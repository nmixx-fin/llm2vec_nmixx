import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
# Enable anomaly detection to pinpoint any in-place operations during backward
torch.autograd.set_detect_anomaly(True)


from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset, Dataset as HuggingFaceDataset

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


#############################################
#   Argument Data Classes
#############################################

@dataclass
class ModelArguments:
    """
    모델 관련 인자: base model 경로, torch_dtype 등을 지정합니다.
    """
    model_name_or_path: str = field(
        default="sentence-transformers/all-MiniLM-L12-v2",
        metadata={"help": "Pretrained model identifier from huggingface.co/models or local path"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default `torch.dtype`. Options: 'auto', 'float16', 'bfloat16', 'float32'."
        },
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
    """
    데이터셋 및 전처리 관련 인자
    """
    dataset_name: Optional[str] = field(
        default="nmixx-fin/NMIXX_train",
        metadata={"help": "The name of the dataset to use from Hugging Face Hub"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use."}
    )
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the train split in the dataset"}
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
    instruction_text: Optional[str] = field(
        default="",
        metadata={"help": "Instruction text to prepend to source texts. Not typically used for MiniLM."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging or quicker training, limit the number of training examples"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    trust_remote_code_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading dataset."}
    )


@dataclass
class CustomTrainingArguments:
    """
    Triplet Loss 관련 커스텀 인자
    """
    margin: float = field(
        default=1.0, metadata={"help": "Margin for TripletMarginLoss"}
    )


#############################################
#   Model & Pooling Module
#############################################

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        last_hidden_state: (batch_size, seq_len, hidden_dim)
        attention_mask:     (batch_size, seq_len) - torch.LongTensor
        """
        # 1) attention_mask을 float로 변환하고 차원을 늘려서 브로드캐스트만 사용
        mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        # 2) 브로드캐스팅: last_hidden_state * mask → (B, S, H)
        masked_embeddings = last_hidden_state * mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # (B, H)
        # 3) mask.sum(dim=1): (B, 1) → 클램프 후 나누기
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return sum_embeddings / sum_mask  # (B, H)


class SentenceTransformerModel(nn.Module):
    def __init__(self, model_name_or_path: str, model_args: ModelArguments):
        super().__init__()
        self.model_args = model_args

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
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask):
        # compute_loss 단계에서 이미 .to(device).clone() 된 텐서들이 들어오므로
        # 여기서는 추가 복제가 필요 없지만, 안전을 위해 한번 더 클론합니다.
        input_ids = input_ids.clone()
        attention_mask = attention_mask.clone()

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state  # (B, S, H)

        pooled = self.pooling(last_hidden_state, attention_mask)
        return pooled

    def save_pretrained(self, save_directory: str):
        logger.info(f"Saving SentenceTransformerModel to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        with open(
            os.path.join(save_directory, "sentence_transformer_model_args.json"),
            "w",
        ) as f:
            json.dump(self.model_args.__dict__, f, indent=4)


#############################################
#   Data Collator & Dataset
#############################################

class TripletDataCollator:
    def __init__(self, tokenizer, max_length: int, instruction: str,
                 source_col: str, positive_col: str, negative_col: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction.strip() if instruction else ""
        self.source_col = source_col
        self.positive_col = positive_col
        self.negative_col = negative_col

    def _prepare_text(self, text: str, with_instruction: bool = False) -> str:
        processed_text = text.strip()
        if with_instruction and self.instruction:
            return f"{self.instruction} {processed_text}"
        return processed_text

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        source_texts = [self._prepare_text(str(ex[self.source_col]), with_instruction=True) for ex in features]
        positive_texts = [self._prepare_text(str(ex[self.positive_col])) for ex in features]
        negative_texts = [self._prepare_text(str(ex[self.negative_col])) for ex in features]

        batch = {
            "source": self.tokenizer(
                source_texts,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt"
            ),
            "positive": self.tokenizer(
                positive_texts,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt"
            ),
            "negative": self.tokenizer(
                negative_texts,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt"
            ),
        }
        return batch


class TripletDataset(TorchDataset):
    def __init__(self, data_args: DataTrainingArguments, cache_dir: Optional[str] = None):
        self.data_args = data_args
        self.samples = []

        logger.info(f"Loading data from Hugging Face dataset: {data_args.dataset_name}")
        raw_dataset: HuggingFaceDataset = load_dataset(
            data_args.dataset_name,
            name=data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=cache_dir,
            trust_remote_code=data_args.trust_remote_code_dataset,
        )
        for i in range(len(raw_dataset)):
            self.samples.append(raw_dataset[i])

        if not self.samples:
            raise ValueError("No data loaded.")

        required_cols = [
            data_args.source_column,
            data_args.positive_column,
            data_args.negative_column,
        ]
        if not all(col in self.samples[0] for col in required_cols):
            raise ValueError(
                f"Dataset items must contain keys: {', '.join(required_cols)}. "
                f"Found: {list(self.samples[0].keys())}"
            )

        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            self.samples = self.samples[: min(data_args.max_train_samples, len(self.samples))]

        logger.info(f"Loaded {len(self.samples)} training examples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, str]:
        sample = self.samples[idx]
        return {
            self.data_args.source_column: sample[self.data_args.source_column],
            self.data_args.positive_column: sample[self.data_args.positive_column],
            self.data_args.negative_column: sample[self.data_args.negative_column],
        }


#############################################
#   Custom Trainer
#############################################

class SentenceTransformerTripletTrainer(Trainer):
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device

        # GPU에 올린 뒤 즉시 clone() 처리해 완전히 분리된 텐서를 사용
        src_ids = inputs["source"]["input_ids"].to(device).clone()
        src_mask = inputs["source"]["attention_mask"].to(device).clone()
        pos_ids = inputs["positive"]["input_ids"].to(device).clone()
        pos_mask = inputs["positive"]["attention_mask"].to(device).clone()
        neg_ids = inputs["negative"]["input_ids"].to(device).clone()
        neg_mask = inputs["negative"]["attention_mask"].to(device).clone()

        source_emb = model(input_ids=src_ids, attention_mask=src_mask)
        positive_emb = model(input_ids=pos_ids, attention_mask=pos_mask)
        negative_emb = model(input_ids=neg_ids, attention_mask=neg_mask)

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
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(output_dir)
        else:
            logger.warning("Model does not have 'save_pretrained'. Saving state_dict directly.")
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info(f"Full model and tokenizer saved to {output_dir}")


#############################################
#   Main function
#############################################

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(f"Model arguments:\n{model_args}")
    logger.info(f"Data arguments:\n{data_args}")
    logger.info(f"Custom training arguments:\n{custom_args}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code
    )

    model = SentenceTransformerModel(model_args.model_name_or_path, model_args)
    logger.info(f"Loaded SentenceTransformerModel with base model '{model_args.model_name_or_path}'.")

    train_dataset = TripletDataset(data_args, cache_dir=model_args.cache_dir)
    data_collator = TripletDataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length or 512,
        instruction=data_args.instruction_text or "",
        source_col=data_args.source_column,
        positive_col=data_args.positive_column,
        negative_col=data_args.negative_column,
    )

    trainer = SentenceTransformerTripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        margin=custom_args.margin,
    )

    try:
        logger.info("Starting model training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")

        final_save_path = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_save_path)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
