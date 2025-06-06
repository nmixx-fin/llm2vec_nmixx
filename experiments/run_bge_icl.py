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
from datasets import load_dataset, Dataset as HuggingFaceDataset

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# ────────────────────────────────────────────────────────────────────────────────
# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# in‐place 연산 오류를 추적하고 싶을 때만 활성화
# torch.autograd.set_detect_anomaly(True)
# ────────────────────────────────────────────────────────────────────────────────


#############################################
#   Argument Data Classes
#############################################
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="BAAI/bge-en-icl",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Override the default `torch.dtype`. Options: 'auto', 'bfloat16', 'float16', 'float32'."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading model/tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="nmixx-fin/NMIXX_train",
        metadata={"help": "The name of the dataset to use from Hugging Face Hub"}
    )
    source_column: str = field(
        default="source_text",
        metadata={"help": "The name of the column containing the source/anchor texts"}
    )
    positive_column: str = field(
        default="positive",
        metadata={"help": "The name of the column containing the positive texts"}
    )
    negative_column: str = field(
        default="negative",
        metadata={"help": "The name of the column containing the negative texts"}
    )
    instruction_text: str = field(
        default="",  # 빈 문자열 허용
        metadata={"help": "Instruction text to prepend to source texts (optional)."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging, limit the number of training examples"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )


@dataclass
class CustomTrainingArguments:
    margin: float = field(
        default=1.0, metadata={"help": "Margin for TripletMarginLoss"}
    )
    lora_r: int = field(
        default=16, metadata={"help": "LoRA rank. Set to 0 or less to disable LoRA."}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Modules to apply LoRA to."}
    )


#############################################
#   Model & Pooling Module
#############################################
class BGEModelWrapper(nn.Module):
    def __init__(self, model_name_or_path: str, model_args: ModelArguments):
        super().__init__()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto"
        }
        torch_dtype_value = dtype_map.get(str(model_args.torch_dtype).lower(), "auto")

        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype_value,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
        )

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Padding이 좌우 어느 쪽이든 상관없이 sum(dim=1)-1이 마지막 유효 토큰 index
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = self._last_token_pool(outputs.last_hidden_state, attention_mask)
        # BGE-ICL 예시처럼 L2 정규화
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def save_pretrained(self, save_directory: str):
        """
        LoRA를 사용중이라면 adapter 가중치를 base model에 병합(merge) 후 저장,
        그렇지 않으면 base model 그대로 저장합니다.
        """
        os.makedirs(save_directory, exist_ok=True)
        logger.info(f"Saving model (merged if LoRA) to {save_directory}")

        # LoRA가 적용된 상태라면 self.base_model이 PeftModel 타입
        if isinstance(self.base_model, PeftModel):
            # merge_and_unload()는 adapter 가중치를 base 모델에 병합하고
            # PeftModel 래퍼를 벗겨낸 순수 base 모델을 반환합니다.
            merged_model = self.base_model.merge_and_unload()
            # merged_model은 nn.Module (base 모델) 형태이므로, save_pretrained 호출 가능
            merged_model.save_pretrained(save_directory)
        else:
            # LoRA가 적용되지 않은 일반 base 모델을 저장
            self.base_model.save_pretrained(save_directory)


#############################################
#   Data Collator & Dataset
#############################################
class TripletDataset(TorchDataset):
    def __init__(self, data_args: DataTrainingArguments, cache_dir: Optional[str] = None):
        set_seed(42)  # 데이터 로드 전에 랜덤 시드 고정
        logger.info(f"Loading data from Hugging Face dataset: {data_args.dataset_name}")
        raw_dataset = load_dataset(data_args.dataset_name, split="train", cache_dir=cache_dir)
        # 컬럼 존재 여부 검증
        first = raw_dataset[0]
        required_cols = [
            data_args.source_column,
            data_args.positive_column,
            data_args.negative_column,
        ]
        if not all(col in first for col in required_cols):
            raise ValueError(
                f"Dataset must contain columns {required_cols}, but found {list(first.keys())}"
            )

        self.samples = list(raw_dataset)
        if data_args.max_train_samples:
            self.samples = self.samples[: data_args.max_train_samples]
        logger.info(f"Loaded {len(self.samples)} training examples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, str]:
        return self.samples[idx]


class TripletDataCollatorForBGE:
    def __init__(self, tokenizer, data_args: DataTrainingArguments):
        self.tokenizer = tokenizer
        self.max_length = data_args.max_seq_length
        self.instruction = data_args.instruction_text.strip()
        self.source_col = data_args.source_column
        self.positive_col = data_args.positive_column
        self.negative_col = data_args.negative_column

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        # instruction이 비어있지 않을 때만 태그 추가
        if self.instruction:
            source_texts = [
                f"<instruct> {self.instruction}\n<query> {ex[self.source_col]}"
                for ex in features
            ]
        else:
            source_texts = [ex[self.source_col] for ex in features]

        positive_texts = [ex[self.positive_col] for ex in features]
        negative_texts = [ex[self.negative_col] for ex in features]

        batch_source = self.tokenizer(
            source_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch_positive = self.tokenizer(
            positive_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch_negative = self.tokenizer(
            negative_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "source": batch_source,
            "positive": batch_positive,
            "negative": batch_negative
        }


#############################################
#   Custom Trainer (Triplet Loss + LoRA 병합 저장)
#############################################
class BGEMarginTripletTrainer(Trainer):
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device

        # GPU로 옮긴 후 즉시 clone() 하여 in-place 이슈 방지
        src_inputs = {k: v.to(device).clone() for k, v in inputs["source"].items()}
        pos_inputs = {k: v.to(device).clone() for k, v in inputs["positive"].items()}
        neg_inputs = {k: v.to(device).clone() for k, v in inputs["negative"].items()}

        source_emb = model(**src_inputs)
        positive_emb = model(**pos_inputs)
        negative_emb = model(**neg_inputs)

        loss = self.triplet_loss(source_emb, positive_emb, negative_emb)
        if return_outputs:
            return loss, {
                "source_emb": source_emb,
                "positive_emb": positive_emb,
                "negative_emb": negative_emb,
            }
        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # DDP → model.module, single-GPU → model 그대로
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        # ❶ LoRA가 적용된 경우 merge & unload
        if isinstance(model_to_save.base_model, PeftModel):
            logger.info("Merging LoRA adapter into base model before saving...")
            merged_model = model_to_save.base_model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            logger.info("Saving base model (no LoRA present)...")
            model_to_save.base_model.save_pretrained(output_dir)

        # ❷ 토크나이저도 함께 저장
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # ❸ TrainingArguments 저장
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


#############################################
#   Main function
#############################################
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info(f"Model arguments:\n{model_args}")
    logger.info(f"Data arguments:\n{data_args}")
    logger.info(f"Training arguments:\n{training_args}")
    logger.info(f"Custom training arguments:\n{custom_args}")

    # 랜덤 시드 설정
    set_seed(training_args.seed)

    # 토크나이저/모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code
    )
    model = BGEModelWrapper(model_args.model_name_or_path, model_args)

    # LoRA 적용 (r > 0 일 때만)
    if custom_args.lora_r > 0:
        logger.info("Applying LoRA to the model.")
        peft_config = LoraConfig(
            r=custom_args.lora_r,
            lora_alpha=custom_args.lora_alpha,
            lora_dropout=custom_args.lora_dropout,
            target_modules=custom_args.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # BGE-ICL이 Causal LM 계열이라면
        )
        model.base_model = get_peft_model(model.base_model, peft_config)
        model.base_model.print_trainable_parameters()

    # 데이터셋 & 데이터콜레이터
    train_dataset = TripletDataset(data_args, cache_dir=model_args.cache_dir)
    data_collator = TripletDataCollatorForBGE(tokenizer=tokenizer, data_args=data_args)

    # Trainer 설정
    trainer = BGEMarginTripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        margin=custom_args.margin,
    )

    logger.info("Starting model training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.info("Training finished. Saving final model...")

    # 최종 저장 (LoRA가 있을 경우 merge & unload)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
