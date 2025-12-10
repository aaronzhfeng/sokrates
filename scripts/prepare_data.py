#!/usr/bin/env python3
"""
Prepare data for SOKRATES training.

Downloads and processes FOLIO and PrOntoQA datasets,
converting them to the optionized format.

Data sources:
- FOLIO: yale-nlp/FOLIO (gated, requires HuggingFace access)
- PrOntoQA: jzfeng/LoGiPT-data (from NAACL'24 paper "Language Models can be Deductive Solvers")
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.optionizer import Optionizer
from src.data.structures import LogicalState, FOLFormula

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_folio(output_dir: str) -> Path | None:
    """
    Download and process FOLIO dataset from HuggingFace.
    
    Source: yale-nlp/FOLIO (gated dataset, requires access request)
    https://huggingface.co/datasets/yale-nlp/FOLIO
    
    Returns:
        Path to output directory if successful, None otherwise.
    """
    logger.info("Downloading FOLIO from yale-nlp/FOLIO...")
    
    try:
        dataset = load_dataset("yale-nlp/FOLIO")
    except Exception as e:
        logger.error(f"Failed to download FOLIO: {e}")
        logger.error("")
        logger.error("FOLIO is a GATED dataset. To access it:")
        logger.error("  1. Visit https://huggingface.co/datasets/yale-nlp/FOLIO")
        logger.error("  2. Request access (requires HuggingFace login)")
        logger.error("  3. Run: huggingface-cli login")
        logger.error("  4. Re-run this script")
        logger.error("")
        logger.error("Alternative: Download manually from https://github.com/Yale-LILY/FOLIO")
        return None
    
    output_path = Path(output_dir) / "folio"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "validation"]:
        if split not in dataset:
            logger.warning(f"FOLIO split '{split}' not found, skipping")
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
        output_file = output_path / f"{split}.json"
        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} FOLIO {split} examples to {output_file}")
    
    return output_path


def download_prontoqa(output_dir: str, test_split_ratio: float = 0.1) -> Path | None:
    """
    Download and process PrOntoQA dataset from HuggingFace.
    
    Source: jzfeng/LoGiPT-data (from NAACL'24 paper "Language Models can be Deductive Solvers")
    https://huggingface.co/datasets/jzfeng/LoGiPT-data
    
    The dataset uses a conversation format that we parse to extract:
    - Context: logical facts and rules
    - Query: the question to answer
    - Chain: reasoning steps (extracted from GPT response)
    - Answer: True/False
    
    Args:
        output_dir: Directory to save raw data
        test_split_ratio: Fraction to use as test set (default 0.1)
    
    Returns:
        Path to output directory if successful, None otherwise.
    """
    import random
    import re
    
    logger.info("Downloading PrOntoQA from jzfeng/LoGiPT-data...")
    
    try:
        dataset = load_dataset("jzfeng/LoGiPT-data")
    except Exception as e:
        logger.error(f"Failed to download PrOntoQA from jzfeng/LoGiPT-data: {e}")
        logger.error("")
        logger.error("Please check:")
        logger.error("  1. Your internet connection")
        logger.error("  2. HuggingFace availability")
        logger.error("  3. Try: pip install --upgrade datasets")
        logger.error("")
        logger.error("Dataset info: https://huggingface.co/datasets/jzfeng/LoGiPT-data")
        return None
    
    output_path = Path(output_dir) / "prontoqa"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Available splits: {list(dataset.keys())}")
    
    # Process the train split (only split available)
    if "train" not in dataset:
        logger.error("No 'train' split found in dataset")
        return None
    
    examples = []
    skipped = 0
    
    for item in tqdm(dataset["train"], desc="Processing PrOntoQA"):
        parsed = parse_logipt_conversation(item)
        if parsed:
            examples.append(parsed)
        else:
            skipped += 1
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} examples that couldn't be parsed")
    
    logger.info(f"Parsed {len(examples)} PrOntoQA examples")
    
    # Split into train/test
    random.seed(42)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * (1 - test_split_ratio))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Save train
    train_file = output_path / "train.json"
    with open(train_file, "w") as f:
        json.dump(train_examples, f, indent=2)
    logger.info(f"Saved {len(train_examples)} PrOntoQA train examples to {train_file}")
    
    # Save test
    test_file = output_path / "test.json"
    with open(test_file, "w") as f:
        json.dump(test_examples, f, indent=2)
    logger.info(f"Saved {len(test_examples)} PrOntoQA test examples to {test_file}")
    
    return output_path


def parse_logipt_conversation(item: dict) -> dict | None:
    """
    Parse a LoGiPT conversation format item into our standard format.
    
    The conversation format has:
    - conversations[0]: human - context with logical facts/rules
    - conversations[1]: gpt - predicate definitions and reasoning
    - conversations[2]: human - question
    - conversations[3]: gpt - answer
    """
    import re
    
    try:
        conversations = item.get("conversations", [])
        if len(conversations) < 4:
            return None
        
        # Extract context from first human message
        first_human = conversations[0].get("value", "")
        context_match = re.search(r"Context:\s*\n(.+?)\n\nReasoning:", first_human, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        
        # Extract query from third human message  
        third_human = conversations[2].get("value", "")
        query_match = re.search(r"is the following comment true or false\?\s*(.+?)\n", third_human, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""
        
        # Extract answer from fourth gpt message
        fourth_gpt = conversations[3].get("value", "")
        answer = "A) True" in fourth_gpt or "true" in fourth_gpt.lower().split("correct option is")[-1][:20]
        
        # Extract reasoning chain from second gpt message
        second_gpt = conversations[1].get("value", "")
        chain = extract_reasoning_chain(second_gpt)
        
        if not context or not query:
            return None
        
        return {
            "id": item.get("id", ""),
            "context": context,
            "query": query,
            "chain": chain,
            "answer": answer,
        }
        
    except Exception as e:
        return None


def extract_reasoning_chain(gpt_response: str) -> list[str]:
    """Extract reasoning steps from GPT response."""
    import re
    
    chain = []
    
    # Extract "Obtain a new implied fact: ..." lines
    facts = re.findall(r"Obtain a new implied fact:\s*(.+?)(?:\n|$)", gpt_response)
    for fact in facts:
        # Convert "Sweet('Fae', True)" to more readable form
        readable = fact.strip()
        chain.append(readable)
    
    return chain


def create_optionized_data(raw_dir: str, output_dir: str) -> None:
    """Convert raw data to optionized format for training."""
    logger.info("Creating optionized data...")
    
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
                # FOLIO premises can be a newline-separated string or list
                raw_premises = ex.get("premises", [])
                if isinstance(raw_premises, str):
                    premises_list = [p.strip() for p in raw_premises.split("\n") if p.strip()]
                else:
                    premises_list = raw_premises
                
                # Same for FOL premises (usually empty in FOLIO)
                raw_premises_fol = ex.get("premises_fol", [])
                if isinstance(raw_premises_fol, str):
                    premises_fol_list = [p.strip() for p in raw_premises_fol.split("\n") if p.strip()]
                else:
                    premises_fol_list = raw_premises_fol if raw_premises_fol else []
                
                # Pad FOL premises if shorter than NL premises
                while len(premises_fol_list) < len(premises_list):
                    premises_fol_list.append("")
                
                # Create LogicalState
                formulas = []
                for i, (nl, fol) in enumerate(zip(premises_list, premises_fol_list)):
                    formulas.append(FOLFormula(
                        id=i,
                        nl_text=nl,
                        fol_string=fol if fol else "",
                        source="premise",
                    ))
                
                state = LogicalState(
                    problem_id=ex["id"],
                    nl_premises=premises_list,
                    fol_formulas=formulas,
                    target_conclusion=ex.get("conclusion", ""),
                    label=ex.get("label", "UNKNOWN").upper(),
                )
                
                optionized.append({
                    "problem_id": state.problem_id,
                    "prompt": state.to_prompt(),
                    "label": state.label,
                    "premises": premises_list,  # Now a proper list
                    "premises_fol": premises_fol_list,
                    "conclusion": ex.get("conclusion", ""),
                    "conclusion_fol": ex.get("conclusion_fol", ""),
                })
            
            output_file = output_path / f"folio_{split_name}.jsonl"
            with open(output_file, "w") as f:
                for item in optionized:
                    f.write(json.dumps(item) + "\n")
            
            logger.info(f"Saved {len(optionized)} optionized FOLIO {split_name} examples to {output_file}")
    else:
        logger.warning(f"FOLIO raw data not found at {folio_path}")
    
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
                
                # Parse context into premises for structured output
                premises = [s.strip() for s in ex["context"].split(".") if s.strip()]
                
                optionized.append({
                    "problem_id": trace.problem_id,
                    "training_text": trace.to_training_string(),
                    "label": trace.final_answer,
                    "num_steps": trace.num_steps,
                    # Structured fields for trace generation
                    "premises": premises,
                    "conclusion": ex["query"],
                })
            
            output_file = output_path / f"prontoqa_{split_name}.jsonl"
            with open(output_file, "w") as f:
                for item in optionized:
                    f.write(json.dumps(item) + "\n")
            
            logger.info(f"Saved {len(optionized)} optionized PrOntoQA {split_name} examples to {output_file}")
    else:
        logger.warning(f"PrOntoQA raw data not found at {prontoqa_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SOKRATES training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  FOLIO:    yale-nlp/FOLIO (gated - request access first)
  PrOntoQA: jzfeng/LoGiPT-data (public)

Examples:
  # Download all datasets
  python scripts/prepare_data.py

  # Download only FOLIO
  python scripts/prepare_data.py --folio-only

  # Download only PrOntoQA  
  python scripts/prepare_data.py --prontoqa-only

  # Skip download, just process existing raw data
  python scripts/prepare_data.py --skip-download
        """
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only process existing raw data",
    )
    parser.add_argument(
        "--folio-only",
        action="store_true",
        help="Only download/process FOLIO dataset",
    )
    parser.add_argument(
        "--prontoqa-only",
        action="store_true",
        help="Only download/process PrOntoQA dataset",
    )
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track success
    folio_success = False
    prontoqa_success = False
    
    # Download datasets
    if not args.skip_download:
        if not args.prontoqa_only:
            result = download_folio(args.raw_dir)
            folio_success = result is not None
            if not folio_success:
                logger.warning("FOLIO download failed - continuing without FOLIO data")
        
        if not args.folio_only:
            result = download_prontoqa(args.raw_dir)
            prontoqa_success = result is not None
            if not prontoqa_success:
                logger.warning("PrOntoQA download failed - continuing without PrOntoQA data")
    else:
        # Check if raw data exists
        folio_success = (Path(args.raw_dir) / "folio").exists()
        prontoqa_success = (Path(args.raw_dir) / "prontoqa").exists()
    
    # Create optionized data
    create_optionized_data(args.raw_dir, args.output_dir)
    
    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Data preparation complete!")
    logger.info(f"  Raw data:       {args.raw_dir}")
    logger.info(f"  Processed data: {args.output_dir}")
    logger.info(f"  FOLIO:          {'✓' if folio_success else '✗ (failed or skipped)'}")
    logger.info(f"  PrOntoQA:       {'✓' if prontoqa_success else '✗ (failed or skipped)'}")
    logger.info("=" * 50)
    
    if not folio_success and not prontoqa_success and not args.skip_download:
        logger.error("No datasets were downloaded successfully!")
        sys.exit(1)


if __name__ == "__main__":
    main()
