# lunafixer_trainer.py
# Production-ready Python framework for fine-tuning DeepSeek-Coder-1.3B-Instruct as an Automated Bug Fixing Engine using QLoRA.
# Author: Senior AI Engineer and Software Architect
# Date: March 2026

import os
import logging
import json
import pandas as pd
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lunafixer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    model_name: str = "deepseek-ai/DeepSeek-Coder-1.3B-Instruct"
    dataset_path: str = os.getenv("DATASET_PATH", "bugfix_dataset.json")
    output_dir: str = os.getenv("OUTPUT_DIR", "./lunafixer_checkpoints")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

class LunaFixerTrainer:
    """
    Production-ready class for fine-tuning DeepSeek-Coder-1.3B-Instruct as an Automated Bug Fixing Engine.
    
    Features:
    - 4-bit QLoRA with PEFT LoRA adapters for VRAM optimization
    - Robust data pipeline for JSON/CSV bugfix datasets
    - Optimized TrainingArguments with BF16, gradient accumulation, cosine scheduler
    - Secure inference with safety constraints
    - Plug-and-play for SOC integration via env vars
    - Comprehensive logging and error handling
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_and_preprocess_data(self) -> Dataset:
        """Load JSON/CSV dataset and preprocess into SFT format."""
        try:
            if self.config.dataset_path.endswith('.json'):
                with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif self.config.dataset_path.endswith(('.csv', '.parquet')):
                data = pd.read_csv(self.config.dataset_path).to_dict('records')
            else:
                raise ValueError(f"Unsupported dataset format: {self.config.dataset_path}")

            # Expected format: [{"buggy_code": "...", "fixed_code": "...", "explanation": "..."}]
            def format_example(example: Dict[str, str]) -> Dict[str, str]:
                prompt = f"<|system|>You are LunaFixer, an expert Automated Bug Fixing Engine. Analyze the buggy code, fix all bugs securely without introducing vulnerabilities, and provide the corrected code with explanation.<|end|>
<|user|>
Buggy code:
{example['buggy_code']}
<|end|>
<|assistant|>
"
                completion = f"Fixed code:
```python
{example['fixed_code']}
```
Explanation: {example.get('explanation', 'Bug fixed securely.')}"
                return {"text": prompt + completion}

            processed_data = [format_example(ex) for ex in data]
            dataset = Dataset.from_list(processed_data)
            logger.info(f"Loaded and preprocessed {len(dataset)} examples.")
            return dataset

        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            raise

    def setup_model_and_tokenizer(self):
        """Setup quantized model and tokenizer with LoRA."""
        # Quantization config [web:13][web:11]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer [web:19][web:22]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # "<|end▁of▁sentence|>"
        self.tokenizer.padding_side = "right"

        # Load base model [web:1]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for kbit training [web:6]
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()

        # LoRA config optimized for DeepSeek [web:23][web:14][web:6]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

        # Apply LoRA [web:13]
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("Model and LoRA adapters setup complete.")

    def setup_trainer(self, dataset: Dataset):
        """Setup SFTTrainer with optimized arguments [web:17][web:13]."""
        response_template = "<|assistant|>
"

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            report_to="none",  # Disable external logging for production
            remove_unused_columns=False,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            data_collator=collator,
            packing=True,
            dataset_text_field="text",
        )
        logger.info("Trainer setup complete.")

    def train(self):
        """Execute training loop."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        self.trainer.train()
        self.trainer.save_model(self.config.output_dir)
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")

    def inference(self, buggy_code: str, adapter_path: Optional[str] = None, max_new_tokens: int = 512) -> str:
        """
        Perform real-time bug fixing inference.
        
        Args:
            buggy_code: Raw buggy code input
            adapter_path: Path to trained LoRA adapters (optional if loaded)
            max_new_tokens: Max tokens to generate
        
        Returns:
            Fixed code + explanation
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Load adapters if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"Loaded adapters from {adapter_path}")

        prompt = f"<|system|>You are LunaFixer, an expert Automated Bug Fixing Engine. Fix the bugs securely without introducing vulnerabilities (no eval/exec, validate inputs, use safe practices). Provide fixed code and explanation.<|end|>
<|user|>
Buggy code:
{buggy_code}
<|end|>
<|assistant|>
"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info("Inference complete.")
        return response.strip()

def main():
    """Example usage for training."""
    config = TrainingConfig()
    trainer = LunaFixerTrainer(config)
    dataset = trainer.load_and_preprocess_data()
    trainer.setup_model_and_tokenizer()
    trainer.setup_trainer(dataset)
    trainer.train()

if __name__ == "__main__":
    main()
