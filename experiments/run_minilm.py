import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset # Renamed to avoid conflict with datasets.Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
# `datasets` 라이브러리의 `load_dataset`과 `Dataset`을 임포트합니다.
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
        default="sentence-transformers/all-MiniLM-L12-v2", # MiniLM 기본 모델로 변경
        metadata={"help": "Pretrained model identifier from huggingface.co/models or local path"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                    "dtype will be automatically derived from the model's weights. Options: 'auto', 'float16', 'bfloat16', 'float32'."
        },
    )
    trust_remote_code: bool = field(
        default=False, # MiniLM은 일반적으로 remote code가 필요 없습니다.
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
        default="nmixx-fin/NMIXX_train", # NMIXX_train 데이터셋으로 변경
        metadata={"help": "The name of the dataset to use from Hugging Face Hub"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the train split in the dataset"}
    )
    source_column: str = field(
        default="source_text", # 데이터셋 이미지에 맞게 컬럼명 변경
        metadata={"help": "The name of the column containing the source/anchor texts"}
    )
    positive_column: str = field(
        default="positive", # 데이터셋 이미지에 맞게 컬럼명 변경
        metadata={"help": "The name of the column containing the positive texts"}
    )
    negative_column: str = field(
        default="negative", # 데이터셋 이미지에 맞게 컬럼명 변경
        metadata={"help": "The name of the column containing the negative texts"}
    )
    instruction_text: Optional[str] = field(
        default="", # MiniLM은 일반적으로 instruction을 사용하지 않으므로 기본값을 비워둡니다.
        metadata={"help": "Instruction text to prepend to source texts. If empty, no instruction is added."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging or quicker training, limit the number of training examples"}
    )
    max_seq_length: Optional[int] = field(
        default=512, # MiniLM의 max_position_embeddings에 맞춰 조정
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    trust_remote_code_dataset: bool = field(
        default=True, # NMIXX_train 데이터셋 로드 시 필요할 수 있음
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
    """
    토큰 임베딩과 attention mask를 받아 mean pooling을 수행하는 모듈.
    """
    def __init__(self):
        super().__init__()

    def forward(self, model_output, attention_mask):
        token_embeddings = model_output[0] # Last hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

class SentenceTransformerModel(nn.Module):
    """
    기본 모델과 mean pooling layer를 결합한 SentenceTransformer 모델.
    """
    def __init__(self, model_name_or_path: str, model_args: ModelArguments):
        super().__init__()
        self.model_args = model_args

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}
        torch_dtype_value = dtype_map.get(str(model_args.torch_dtype).lower(), "auto")

        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype_value,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
        )
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooling(outputs, attention_mask)
        return pooled

    def save_pretrained(self, save_directory: str):
        """Saves the base model and the pooling configuration (if any)."""
        logger.info(f"Saving SentenceTransformerModel to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        # ModelArguments를 저장하여 모델 래퍼의 설정을 보존합니다.
        with open(os.path.join(save_directory, "sentence_transformer_model_args.json"), "w") as f:
            json.dump(self.model_args.__dict__, f, indent=4)


#############################################
#   Data Collator & Dataset
#############################################

class TripletDataCollator:
    """ Triplet 데이터를 토큰화하고 배치로 만드는 데이터 콜레이터 """
    def __init__(self, tokenizer, max_length: int, instruction: str,
                 source_col: str, positive_col: str, negative_col: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction.strip() if instruction else ""
        self.source_col = source_col
        self.positive_col = positive_col
        self.negative_col = negative_col

    def _prepare_text_with_instruction(self, text: str) -> str:
        """Instruction을 텍스트 앞에 결합합니다."""
        if self.instruction:
            return f"{self.instruction} {text.strip()}"
        return text.strip()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        source_texts = [self._prepare_text_with_instruction(str(example[self.source_col])) for example in features]
        positive_texts = [str(example[self.positive_col]).strip() for example in features]
        negative_texts = [str(example[self.negative_col]).strip() for example in features]

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
    """
    Loads data from Hugging Face dataset for triplet loss training.
    Each item should have source_column, positive_column, and negative_column.
    """
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
        # Convert to list of dicts
        for i in range(len(raw_dataset)):
             self.samples.append(raw_dataset[i])

        if not self.samples:
            raise ValueError("No data loaded. Please check dataset_name.")

        required_cols = [data_args.source_column, data_args.positive_column, data_args.negative_column]
        if not all(col in self.samples[0] for col in required_cols):
            raise ValueError(f"Dataset items must contain keys: {', '.join(required_cols)}. Found: {list(self.samples[0].keys())}")

        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            logger.info(f"Selecting a subset of {data_args.max_train_samples} training samples.")
            self.samples = self.samples[:min(data_args.max_train_samples, len(self.samples))]

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
    """
    Trainer를 상속받아 (source, positive, negative) 데이터를 받아 TripletMarginLoss를 계산합니다.
    """
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False):
        source_enc = inputs["source"]
        positive_enc = inputs["positive"]
        negative_enc = inputs["negative"]

        device = next(model.parameters()).device

        source_emb = model(
            input_ids=source_enc["input_ids"].to(device),
            attention_mask=source_enc["attention_mask"].to(device)
        )
        positive_emb = model(
            input_ids=positive_enc["input_ids"].to(device),
            attention_mask=positive_enc["attention_mask"].to(device)
        )
        negative_emb = model(
            input_ids=negative_enc["input_ids"].to(device),
            attention_mask=negative_enc["attention_mask"].to(device)
        )

        loss = self.triplet_loss(source_emb, positive_emb, negative_emb)

        if return_outputs:
            return loss, {"source_emb": source_emb, "positive_emb": positive_emb, "negative_emb": negative_emb}
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
            logger.warning("Model object or its 'module' does not have 'save_pretrained'. Saving state_dict directly.")
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
        logger.info(f"Loading arguments from JSON config file: {sys.argv[1]}")
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        logger.info("Loading arguments from command line.")
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
                f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits ({training_args.fp16_backend if training_args.fp16 else 'None'}): {training_args.fp16 or training_args.bf16}")
    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(f"Model arguments:\n{model_args}")
    logger.info(f"Data arguments:\n{data_args}")
    logger.info(f"Custom training arguments:\n{custom_args}")

    set_seed(training_args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code
        )
    except Exception as e:
        logger.error(f"Error loading tokenizer from '{model_args.model_name_or_path}': {e}", exc_info=True)
        sys.exit(1)

    try:
        model = SentenceTransformerModel(model_args.model_name_or_path, model_args)
        logger.info(f"Loaded SentenceTransformerModel with base model '{model_args.model_name_or_path}'.")
    except Exception as e:
        logger.error(f"Error loading SentenceTransformerModel with base '{model_args.model_name_or_path}': {e}", exc_info=True)
        sys.exit(1)


    try:
        train_dataset = TripletDataset(data_args, cache_dir=model_args.cache_dir)
    except Exception as e:
        logger.error(f"Error loading or processing dataset: {e}", exc_info=True)
        sys.exit(1)

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
        margin=custom_args.margin
    )

    try:
        logger.info("Starting model training...")
        if training_args.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")

        final_save_path = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_save_path)

        metrics = train_result.metrics
        if data_args.max_train_samples is not None:
            metrics["train_samples"] = min(data_args.max_train_samples, len(train_dataset))
        else:
            metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model state (if applicable)...")
        if training_args.should_save: # Check if saving is enabled by strategy
            interrupted_save_path = os.path.join(training_args.output_dir, "interrupted_checkpoint")
            try:
                trainer.save_model(interrupted_save_path)
                logger.info(f"Model state saved to {interrupted_save_path}")
            except Exception as e_save:
                logger.error(f"Error saving model during interrupt: {e_save}", exc_info=True)
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()