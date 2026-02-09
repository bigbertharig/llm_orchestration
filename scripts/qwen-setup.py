#!/usr/bin/env python3
"""
Download and run Qwen2.5-7B-Instruct locally.
Uses 4-bit quantization to fit on a single GTX 1060 6GB (~3.5GB VRAM usage).

Usage:
    python qwen-setup.py              # Uses GPU 0
    python qwen-setup.py --gpu 1      # Uses GPU 1
    python qwen-setup.py --download   # Download only, no chat
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (0-3)")
    parser.add_argument("--download", action="store_true", help="Download model only")
    args = parser.parse_args()

    print(f"Model: {MODEL_ID}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
        marker = " <--" if i == args.gpu else ""
        print(f"  GPU {i}: {props.name} - {free_mem:.1f}GB free{marker}")

    # 4-bit quantization config - ~3.5GB VRAM for 7B model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nDownloading/loading model to GPU {args.gpu}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if args.download:
        # Download only - load to CPU to verify files
        print("Download-only mode: fetching model files...")
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL_ID)
        print(f"\nModel downloaded to: {path}")
        print("Run without --download to start chatting.")
        return

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": args.gpu},  # Load entire model to specified GPU
        trust_remote_code=True,
    )

    # Show VRAM usage
    allocated = torch.cuda.memory_allocated(args.gpu) / 1024**3
    print(f"VRAM used on GPU {args.gpu}: {allocated:.2f}GB")

    print("\nModel loaded! Starting interactive chat (type 'quit' to exit)\n")

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nQwen: {response}\n")

        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
