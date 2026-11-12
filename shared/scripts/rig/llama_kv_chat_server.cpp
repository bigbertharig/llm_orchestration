#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include "httplib.h"
#include <algorithm>
#include <atomic>
#include <regex>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
extern "C" void llama_kv_chat_select(int first, int second);

struct session {
    std::string id;
    int gpu;
    int cap;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
    std::atomic<int> used{0};
    std::mutex mutex;
    ~session() { if (sampler) llama_sampler_free(sampler); if (ctx) llama_free(ctx); }
};

static void reply(httplib::Response &res, int code, json body) {
    res.status = code;
    res.set_content(body.dump(), "application/json");
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool special) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, special, true);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n, special, true) != n) return {};
    return tokens;
}

static bool feed(session &s, const std::vector<llama_token> &tokens) {
    llama_kv_chat_select(s.gpu, -1);
    for (size_t i = 0; i < tokens.size();) {
        int n = std::min<size_t>(64, tokens.size() - i);
        if (llama_decode(s.ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), n))) return false;
        i += n;
        s.used += n;
    }
    return true;
}

static std::string piece(const llama_vocab *vocab, llama_token token) {
    int n = -llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
    if (n <= 0) return {};
    std::string out(n, '\0');
    int written = llama_token_to_piece(vocab, token, out.data(), n, 0, true);
    if (written < 0) return {};
    out.resize(written);
    return out;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: llama-kv-chat-server MODEL.gguf PORT CONTEXT_CAP\n");
        return 2;
    }
    const int port = std::atoi(argv[2]);
    const int cap = std::atoi(argv[3]);
    if (port < 1024 || port > 65535 || cap < 16384 || cap > 262144) return 2;
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = 0;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::shared_mutex lifecycle;
    std::map<std::string, std::shared_ptr<session>> sessions;
    auto add = [&](const std::string &id, int gpu) -> bool {
        if (id.empty() || id.size() > 64 || !std::regex_match(id, std::regex("[A-Za-z0-9_-]+")) || gpu < 0 || gpu > 5 || sessions.count(id)) return false;
        for (const auto &[_, item] : sessions) if (item->gpu == gpu) return false;
        auto item = std::make_shared<session>();
        item->id = id; item->gpu = gpu; item->cap = cap;
        llama_kv_chat_select(gpu, -1);
        auto cp = llama_context_default_params();
        cp.n_ctx = cap; cp.n_batch = 64; cp.n_ubatch = 8; cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
        item->ctx = llama_init_from_model(model, cp);
        if (!item->ctx) return false;
        item->sampler = llama_sampler_init_greedy();
        sessions.emplace(id, std::move(item));
        std::fprintf(stderr, "CHAT_ATTACHED id=%s gpu=%d cap=%d\n", id.c_str(), gpu, cap);
        return true;
    };
    if (!add("main", 0)) return 4;
    httplib::Server server;
    server.Get("/health", [&](const httplib::Request &, httplib::Response &res) {
        std::shared_lock lock(lifecycle);
        reply(res, 200, {{"status","ok"},{"model_loaded",true},{"chats",sessions.size()}});
    });
    server.Get("/chats", [&](const httplib::Request &, httplib::Response &res) {
        std::shared_lock lock(lifecycle);
        json rows = json::array();
        for (const auto &[id, item] : sessions)
            rows.push_back({{"chat_id",id},{"gpu",item->gpu},{"used_tokens",item->used.load()},{"context_cap",item->cap}});
        reply(res, 200, {{"chats",rows}});
    });
    server.Post("/chats", [&](const httplib::Request &req, httplib::Response &res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { reply(res,400,{{"error","invalid JSON"}}); return; }
        if (!body.contains("chat_id") || !body["chat_id"].is_string() || !body.contains("gpu") || !body["gpu"].is_number_integer()) {
            reply(res,400,{{"error","chat_id string and gpu integer required"}}); return;
        }
        const std::string id = body["chat_id"].get<std::string>();
        const int gpu = body["gpu"].get<int>();
        if (gpu < 1 || gpu > 5) { reply(res,400,{{"error","worker GPU must be 1..5"}}); return; }
        std::unique_lock lock(lifecycle);
        if (!add(id,gpu)) { reply(res,409,{{"error","chat ID or GPU in use, or allocation failed"}}); return; }
        reply(res,201,{{"chat_id",id},{"gpu",gpu},{"context_cap",cap},{"parent","main"}});
    });
    server.Delete(R"(/chats/([A-Za-z0-9_-]+))", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string id = req.matches[1];
        if (id == "main") { reply(res,409,{{"error","main chat cannot be detached"}}); return; }
        std::unique_lock lock(lifecycle);
        if (!sessions.erase(id)) { reply(res,404,{{"error","unknown chat"}}); return; }
        reply(res,200,{{"detached",id}});
    });
    server.Post(R"(/chats/([A-Za-z0-9_-]+)/turns)", [&](const httplib::Request &req, httplib::Response &res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { reply(res,400,{{"error","invalid JSON"}}); return; }
        if (!body.contains("message") || !body["message"].is_string()) {
            reply(res,400,{{"error","message string required"}}); return;
        }
        const std::string message = body["message"].get<std::string>();
        const int max_tokens = body.value("max_tokens",64);
        if (message.empty() || message.size() > 65536 || max_tokens < 1 || max_tokens > 512) {
            reply(res,400,{{"error","message or max_tokens out of range"}}); return;
        }
        std::shared_lock life(lifecycle);
        const std::string id = req.matches[1];
        auto found = sessions.find(id);
        if (found == sessions.end()) { reply(res,404,{{"error","unknown chat"}}); return; }
        auto item = found->second;
        std::lock_guard turn(item->mutex);
        const bool first = item->used.load() == 0;
        const std::string formatted = std::string(first ? "" : "<|im_end|>\n") +
            "<|im_start|>user\n" + message + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto input = tokenize(vocab, formatted, first);
        if (input.empty() || item->used.load() + (int)input.size() + max_tokens >= cap) {
            reply(res,409,{{"error","context full or tokenization failed"}}); return;
        }
        const auto started = clock_type::now();
        if (!feed(*item,input)) { reply(res,500,{{"error","prefill failed"}}); return; }
        const auto prefill_done = clock_type::now();
        std::string output; int generated = 0; bool eog = false;
        for (; generated < max_tokens; ++generated) {
            llama_kv_chat_select(item->gpu,-1);
            llama_token token = llama_sampler_sample(item->sampler,item->ctx,-1);
            if (llama_vocab_is_eog(vocab,token)) { eog = true; break; }
            output += piece(vocab,token);
            if (!feed(*item,{token})) { reply(res,500,{{"error","decode failed"}}); return; }
        }
        const auto end = clock_type::now();
        const double prefill_s = std::chrono::duration<double>(prefill_done-started).count();
        const double decode_s = std::chrono::duration<double>(end-prefill_done).count();
        reply(res,200,{{"chat_id",id},{"gpu",item->gpu},{"content",output},{"generated_tokens",generated},
            {"used_tokens",item->used.load()},{"finish_reason",eog ? "stop" : "length"},
            {"prefill_seconds",prefill_s},{"decode_seconds",decode_s},{"decode_tok_s",decode_s > 0 ? generated/decode_s : 0}});
    });
    std::fprintf(stderr,"KV_CHAT_SERVER_READY port=%d cap=%d\n",port,cap);
    std::fflush(stderr);
    const bool ok = server.listen("127.0.0.1",port);
    sessions.clear(); llama_model_free(model);
    return ok ? 0 : 5;
}
