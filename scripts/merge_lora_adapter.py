#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM compatibility.

vLLM's LoRA Triton kernels don't work on B200 (Blackwell) GPUs.
This script merges the adapter weights into the base model,
creating a standalone model that vLLM can load directly.

Usage:
    python scripts/merge_lora_adapter.py \
        --adapter outputs/sft/latest/final \
        --output outputs/sft/latest/merged
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(adapter_path: str, output_path: str):
    """
    Merge LoRA adapter into base model and save.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    
    # Load adapter config to get base model name
    with open(adapter_path / "adapter_config.json") as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    print(f"Base model: {base_model_name}")
    print(f"LoRA adapter: {adapter_path}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU to save GPU memory
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load and merge LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Save a marker file indicating this is a merged model
    with open(output_path / "merge_info.json", "w") as f:
        json.dump({
            "base_model": base_model_name,
            "adapter_path": str(adapter_path),
            "merged": True,
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("Merge complete!")
    print("="*60)
    print(f"Merged model saved to: {output_path}")
    print(f"\nUsage with vLLM:")
    print(f"  python scripts/generate_traces_vllm.py --model {output_path} ...")
    

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Output path for merged model")
    args = parser.parse_args()
    
    merge_lora_adapter(args.adapter, args.output)


if __name__ == "__main__":
    main()

