"""
Supervised Fine-Tuning (SFT) for SOKRATES.

Trains the base model on optionized proof traces from P-FOLIO
to learn the Thought/Action format.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import Dataset

from src.data.structures import OptionizedTrace


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_peft: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    output_dir: str = "outputs/sft"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


class OptionizedTraceDataset(Dataset):
    """Dataset of optionized traces for SFT."""
    
    def __init__(
        self,
        traces: list[OptionizedTrace],
        tokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize the dataset.
        
        Args:
            traces: List of OptionizedTrace objects
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        """
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.traces)
    
    def __getitem__(self, idx: int) -> dict:
        trace = self.traces[idx]
        
        # Convert trace to training string
        text = trace.to_training_string()
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # For causal LM, labels = input_ids (shifted internally)
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }


def prepare_sft_data(
    traces: list[OptionizedTrace],
    tokenizer,
    train_ratio: float = 0.9,
    max_length: int = 2048,
) -> tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets for SFT.
    
    Args:
        traces: List of optionized traces
        tokenizer: Tokenizer
        train_ratio: Fraction of data for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Shuffle and split
    import random
    traces = traces.copy()
    random.shuffle(traces)
    
    split_idx = int(len(traces) * train_ratio)
    train_traces = traces[:split_idx]
    val_traces = traces[split_idx:]
    
    train_dataset = OptionizedTraceDataset(train_traces, tokenizer, max_length)
    val_dataset = OptionizedTraceDataset(val_traces, tokenizer, max_length)
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer(config: SFTConfig):
    """
    Set up the model and tokenizer for training.
    
    Args:
        config: SFT configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if running in distributed mode (accelerate sets these env vars)
    is_distributed = (
        os.environ.get("WORLD_SIZE") is not None and 
        int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    
    # Load model
    # Don't use device_map="auto" in distributed mode - accelerate handles device placement
    if is_distributed:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto",
        )
    
    # Apply LoRA if enabled
    if config.use_peft:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing AFTER applying LoRA
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    return model, tokenizer


def train_sft(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[SFTConfig] = None,
):
    """
    Run SFT training.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Training configuration
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    if config is None:
        config = SFTConfig()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",  # renamed from evaluation_strategy
        save_total_limit=3,
        bf16=config.bf16,
        report_to="none",  # Use our custom logging; set to "wandb" if --wandb flag used
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final"))
    
    return trainer


def run_sft_pipeline(
    traces: list[OptionizedTrace],
    config: Optional[SFTConfig] = None,
) -> tuple:
    """
    Run the complete SFT pipeline.
    
    Args:
        traces: List of optionized traces for training
        config: SFT configuration
        
    Returns:
        Tuple of (trained_model, tokenizer, trainer)
    """
    if config is None:
        config = SFTConfig()
    
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_sft_data(
        traces, tokenizer,
        max_length=config.max_seq_length,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    print("Starting SFT training...")
    trainer = train_sft(
        model, tokenizer,
        train_dataset, val_dataset,
        config,
    )
    
    return model, tokenizer, trainer

