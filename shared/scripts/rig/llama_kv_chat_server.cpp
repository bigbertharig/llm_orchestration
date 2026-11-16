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
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
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
    if (argc != 5) {
        std::fprintf(stderr, "usage: llama-kv-chat-server MODEL.gguf PORT CONTEXT_CAP q4_0|f16\n");
        return 2;
    }
    const int port = std::atoi(argv[2]);
    const int cap = std::atoi(argv[3]);
    if (port < 1024 || port > 65535 || cap < 16384 || cap > 262144) return 2;
    const std::string kv_type = argv[4];
    if (kv_type != "q4_0" && kv_type != "f16") return 2;
    const bool nemo = kv_type == "f16";
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = 0;
    std::vector<ggml_backend_dev_t> all_gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) all_gpus.push_back(dev);
    }
    std::vector<ggml_backend_dev_t> model_devices;
    std::vector<float> model_split(llama_max_devices(), 0.0f);
    std::deque<std::string> override_patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (nemo) {
        if (all_gpus.size() < 6) { std::fprintf(stderr, "Nemo needs six visible GPUs\n"); return 6; }
        for (int gpu = 0; gpu < 5; ++gpu) model_devices.push_back(all_gpus[gpu]);
        model_devices.push_back(nullptr);
        model_split[0] = 88.0f;
        mp.devices = model_devices.data();
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        mp.tensor_split = model_split.data();
        const char * path = std::getenv("KV_TENSOR_OVERRIDES_FILE");
        if (!path || !*path) { std::fprintf(stderr, "Nemo override path missing\n"); return 6; }
        std::ifstream input(path);
        if (!input) { std::fprintf(stderr, "Nemo override file cannot be opened: %s\n", path); return 6; }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            const size_t tab = line.find('\t');
            if (tab == std::string::npos || tab + 1 == line.size()) return 6;
            int gpu = -1;
            try { gpu = std::stoi(line.substr(0, tab)); } catch (...) { return 6; }
            if (gpu < 1 || gpu > 4) return 6;
            override_patterns.push_back(line.substr(tab + 1));
            overrides.push_back({override_patterns.back().c_str(), ggml_backend_dev_buffer_type(all_gpus[gpu])});
        }
        if (overrides.empty()) return 6;
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();
    }
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::shared_mutex lifecycle;
    std::map<std::string, std::shared_ptr<session>> sessions;
    auto add = [&](const std::string &id, int gpu) -> bool {
        if (id.empty() || id.size() > 64 || !std::regex_match(id, std::regex("[A-Za-z0-9_-]+")) || gpu < 0 || gpu > 5 || (nemo && gpu != 0 && gpu != 5) || sessions.count(id)) return false;
        for (const auto &[_, item] : sessions) if (item->gpu == gpu) return false;
        auto item = std::make_shared<session>();
        item->id = id; item->gpu = gpu; item->cap = cap;
        llama_kv_chat_select(gpu, -1);
        auto cp = llama_context_default_params();
        cp.n_ctx = cap; cp.n_batch = 64; cp.n_ubatch = 8; cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = nemo ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
        cp.type_v = cp.type_k;
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
        if (gpu < 1 || gpu > 5 || (nemo && gpu != 5)) { reply(res,400,{{"error",nemo ? "Nemo attached chat must use GPU5" : "worker GPU must be 1..5"}}); return; }
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
