import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

# --- 로깅 설정 ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- 인자 클래스 (Arguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "모델 경로 또는 Hugging Face 모델 식별자"}
    )
    torch_dtype: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "데이터셋 이름"})
    source_column: str = field(default="source_text")
    positive_column: str = field(default="positive")
    negative_column: str = field(default="negative")
    instruction_text: str = field(default="", metadata={"help": "Source 텍스트에 추가할 Instruction (작업 설명)"})
    max_train_samples: Optional[int] = field(default=None)
    max_seq_length: int = field(default=512)

@dataclass
class CustomTrainingArguments:
    margin: float = field(default=1.0)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

# --- 모델 및 풀링 모듈 ---
class E5MistralModelWrapper(nn.Module):
    def __init__(self, model_name_or_path: str, model_args: ModelArguments):
        super().__init__()
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}
        torch_dtype_value = dtype_map.get(str(model_args.torch_dtype).lower(), "auto")

        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype_value,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
        )

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = self._last_token_pool(outputs.last_hidden_state, attention_mask)
        return nn.functional.normalize(embeddings, p=2, dim=1)

# --- 데이터셋 및 데이터 콜레이터 ---
class TripletDataset(TorchDataset):
    def __init__(self, data_args: DataTrainingArguments, cache_dir: Optional[str] = None):
        set_seed(42)
        raw_dataset = load_dataset(data_args.dataset_name, split="train", cache_dir=cache_dir)
        self.samples = list(raw_dataset)
        if data_args.max_train_samples:
            self.samples = self.samples[: data_args.max_train_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, str]:
        return self.samples[idx]

class TripletDataCollatorForE5:
    def __init__(self, tokenizer, data_args: DataTrainingArguments):
        self.tokenizer = tokenizer
        self.max_length = data_args.max_seq_length
        self.instruction = data_args.instruction_text.strip()
        self.source_col = data_args.source_column
        self.positive_col = data_args.positive_column
        self.negative_col = data_args.negative_column

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        # E5-Mistral 형식에 맞게 수정
        source_texts = [f"Instruct: {self.instruction}\nQuery: {ex[self.source_col]}" for ex in features]
        positive_texts = [ex[self.positive_col] for ex in features]
        negative_texts = [ex[self.negative_col] for ex in features]

        batch_source = self.tokenizer(source_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        batch_positive = self.tokenizer(positive_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        batch_negative = self.tokenizer(negative_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")

        return {"source": batch_source, "positive": batch_positive, "negative": batch_negative}

# --- 커스텀 트레이너 ---
class E5TripletTrainer(Trainer):
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        src_inputs = {k: v.to(device) for k, v in inputs["source"].items()}
        pos_inputs = {k: v.to(device) for k, v in inputs["positive"].items()}
        neg_inputs = {k: v.to(device) for k, v in inputs["negative"].items()}

        source_emb = model(**src_inputs)
        positive_emb = model(**pos_inputs)
        negative_emb = model(**neg_inputs)

        loss = self.triplet_loss(source_emb, positive_emb, negative_emb)
        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        if isinstance(model_to_save.base_model, PeftModel):
            logger.info("Merging LoRA adapter into base model before saving...")
            merged_model = model_to_save.base_model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            model_to_save.base_model.save_pretrained(output_dir)

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

# --- 메인 함수 ---
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code)
    model = E5MistralModelWrapper(model_args.model_name_or_path, model_args)

    if custom_args.lora_r > 0:
        peft_config = LoraConfig(
            r=custom_args.lora_r,
            lora_alpha=custom_args.lora_alpha,
            lora_dropout=custom_args.lora_dropout,
            target_modules=custom_args.lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model.base_model = get_peft_model(model.base_model, peft_config)
        model.base_model.print_trainable_parameters()

    train_dataset = TripletDataset(data_args, cache_dir=model_args.cache_dir)
    data_collator = TripletDataCollatorForE5(tokenizer=tokenizer, data_args=data_args)

    trainer = E5TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        margin=custom_args.margin,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    main()