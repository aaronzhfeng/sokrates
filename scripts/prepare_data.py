#!/usr/bin/env python3
"""
Prepare data for SOKRATES training.

Downloads and processes FOLIO, P-FOLIO, and PrOntoQA datasets,
converting them to the optionized format.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.optionizer import Optionizer
from src.data.structures import LogicalState, FOLFormula


def download_folio(output_dir: str):
    """Download and process FOLIO dataset."""
    print("Downloading FOLIO...")
    
    # FOLIO is available on HuggingFace
    try:
        dataset = load_dataset("yale-nlp/FOLIO")
    except Exception as e:
        print(f"Could not download FOLIO from HuggingFace: {e}")
        print("Please download manually from https://github.com/Yale-LILY/FOLIO")
        return None
    
    output_path = Path(output_dir) / "folio"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "validation"]:
        if split not in dataset:
            continue
        
        examples = []
        for item in tqdm(dataset[split], desc=f"Processing FOLIO {split}"):
            example = {
                "id": item.get("id", str(len(examples))),
                "premises": item.get("premises", []),
                "premises_fol": item.get("premises_FOL", []),
                "conclusion": item.get("conclusion", ""),
                "conclusion_fol": item.get("conclusion_FOL", ""),
                "label": item.get("label", "Unknown"),
            }
            examples.append(example)
        
        # Save
        with open(output_path / f"{split}.json", "w") as f:
            json.dump(examples, f, indent=2)
        
        print(f"  Saved {len(examples)} FOLIO {split} examples")
    
    return output_path


def download_prontoqa(output_dir: str):
    """Download and process PrOntoQA dataset."""
    print("Downloading PrOntoQA...")
    
    try:
        dataset = load_dataset("rencos/PrOntoQA")
    except Exception as e:
        print(f"Could not download PrOntoQA: {e}")
        return None
    
    output_path = Path(output_dir) / "prontoqa"
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "test"]:
        if split not in dataset:
            continue
        
        examples = []
        for item in tqdm(dataset[split], desc=f"Processing PrOntoQA {split}"):
            example = {
                "id": item.get("id", str(len(examples))),
                "context": item.get("context", ""),
                "query": item.get("query", ""),
                "chain": item.get("chain", []),
                "answer": item.get("answer", True),
            }
            examples.append(example)
        
        with open(output_path / f"{split}.json", "w") as f:
            json.dump(examples, f, indent=2)
        
        print(f"  Saved {len(examples)} PrOntoQA {split} examples")
    
    return output_path


def create_optionized_data(raw_dir: str, output_dir: str):
    """Convert raw data to optionized format."""
    print("\nCreating optionized data...")
    
    optionizer = Optionizer()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process FOLIO
    folio_path = Path(raw_dir) / "folio"
    if folio_path.exists():
        for split_file in folio_path.glob("*.json"):
            split_name = split_file.stem
            
            with open(split_file) as f:
                examples = json.load(f)
            
            optionized = []
            for ex in tqdm(examples, desc=f"Optionizing FOLIO {split_name}"):
                # Create LogicalState
                formulas = []
                for i, (nl, fol) in enumerate(zip(
                    ex.get("premises", []),
                    ex.get("premises_fol", [])
                )):
                    formulas.append(FOLFormula(
                        id=i,
                        nl_text=nl,
                        fol_string=fol if fol else "",
                        source="premise",
                    ))
                
                state = LogicalState(
                    problem_id=ex["id"],
                    nl_premises=ex.get("premises", []),
                    fol_formulas=formulas,
                    target_conclusion=ex.get("conclusion", ""),
                    label=ex.get("label", "UNKNOWN").upper(),
                )
                
                optionized.append({
                    "problem_id": state.problem_id,
                    "prompt": state.to_prompt(),
                    "label": state.label,
                    "premises": ex.get("premises", []),
                    "premises_fol": ex.get("premises_fol", []),
                    "conclusion": ex.get("conclusion", ""),
                    "conclusion_fol": ex.get("conclusion_fol", ""),
                })
            
            with open(output_path / f"folio_{split_name}.jsonl", "w") as f:
                for item in optionized:
                    f.write(json.dumps(item) + "\n")
            
            print(f"  Saved {len(optionized)} optionized FOLIO {split_name} examples")
    
    # Process PrOntoQA
    prontoqa_path = Path(raw_dir) / "prontoqa"
    if prontoqa_path.exists():
        for split_file in prontoqa_path.glob("*.json"):
            split_name = split_file.stem
            
            with open(split_file) as f:
                examples = json.load(f)
            
            optionized = []
            for ex in tqdm(examples, desc=f"Optionizing PrOntoQA {split_name}"):
                trace = optionizer.optionize_prontoqa_example(
                    problem_id=ex["id"],
                    context=ex["context"],
                    query=ex["query"],
                    chain=ex.get("chain", []),
                    label=ex["answer"],
                )
                
                optionized.append({
                    "problem_id": trace.problem_id,
                    "training_text": trace.to_training_string(),
                    "label": trace.final_answer,
                    "num_steps": trace.num_steps,
                })
            
            with open(output_path / f"prontoqa_{split_name}.jsonl", "w") as f:
                for item in optionized:
                    f.write(json.dumps(item) + "\n")
            
            print(f"  Saved {len(optionized)} optionized PrOntoQA {split_name} examples")


def main():
    parser = argparse.ArgumentParser(description="Prepare SOKRATES training data")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw downloaded data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for processed data",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only process existing data",
    )
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download datasets
    if not args.skip_download:
        download_folio(args.raw_dir)
        download_prontoqa(args.raw_dir)
    
    # Create optionized data
    create_optionized_data(args.raw_dir, args.output_dir)
    
    print("\nData preparation complete!")
    print(f"Raw data: {args.raw_dir}")
    print(f"Processed data: {args.output_dir}")


if __name__ == "__main__":
    main()

