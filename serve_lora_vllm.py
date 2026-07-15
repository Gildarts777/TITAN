#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-generate TITAN paths from a locally fine-tuned LoRA adapter, served
through vLLM's native LoRA support (LoRARequest + enable_lora=True) — no
few-shot prompting needed here, unlike baselines/vllm_baseline.py, because
the DSL is fine-tuned into the adapter weights, not specified in-context.

This is the "serve/evaluate via vLLM afterward" half of the decomposed
unsloth-training + vLLM-serving approach (see train_qwen.py's docstring for
why colocated fast_inference=True wasn't usable this session).

Reads the adapter's own adapter_config.json for base_model_name_or_path and
LoRA rank, so it doesn't need to be told what base model was fine-tuned.

Run in .venv-vllm (already proven for the baseline generations):
    source .venv-vllm/bin/activate
    CUDA_VISIBLE_DEVICES=3 python serve_lora_vllm.py \
        --adapter MODELS/phi_titan_nocot_v2 \
        --system-prompt-style plain \
        --test datasets/TEMPLATE_DISJOINT/NoCoT/test_heldout.csv \
        --out baselines/phi_titan_nocot_v2_test_heldout.json
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

SYSTEM_PROMPT = (
    "You are a Cybersecurity Assistant. Generate a valid relation path to answer the following "
    "question or instruction."
)


def read_adapter_config(adapter_dir: str) -> dict:
    path = os.path.join(adapter_dir, "adapter_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM-served LoRA batch generation")
    ap.add_argument("--adapter", required=True, help="directory with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--base-model", default=None,
                    help="override the base model (default: read from adapter_config.json)")
    ap.add_argument("--test", required=True, help="CSV with a Question column")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = read_adapter_config(args.adapter)
    base_model = args.base_model or cfg["base_model_name_or_path"]
    lora_rank = cfg.get("r", 16)
    print(f"[INFO] base model: {base_model} | adapter: {args.adapter} | rank: {lora_rank}")

    test = pd.read_csv(args.test)
    if args.sample and args.sample < len(test):
        total = len(test)
        test = (test.groupby("Section", group_keys=False)
                    .apply(lambda g: g.sample(max(1, round(args.sample * len(g) / total)),
                                              random_state=args.seed))
                    .reset_index(drop=True))
    questions = test["Question"].astype(str).tolist()
    print(f"[INFO] generating for {len(questions)} questions")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model,
        tokenizer=args.adapter,  # use the exact tokenizer/chat_template.jinja saved after training,
                                 # not the base repo's default (checked to match for phi-3.5 but not
                                 # assumed in general)
        quantization="bitsandbytes" if "bnb-4bit" in base_model else None,
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )
    lora_request = LoRARequest("titan_adapter", 1, args.adapter)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                              stop=["</PATH>"], include_stop_str_in_output=True)

    conversations = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": q}]
        for q in questions
    ]

    outputs = llm.chat(conversations, sampling, use_tqdm=True, lora_request=lora_request)

    results = [{"question": q, "generated_path": o.outputs[0].text.strip()}
              for q, o in zip(questions, outputs)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[OK] {len(results)} generations -> {args.out}")


if __name__ == "__main__":
    main()
