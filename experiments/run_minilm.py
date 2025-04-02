#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)


#############################################
#   Argument Data Classes
#############################################

@dataclass
class ModelArguments:
    """
    모델 관련 인자: base model 경로 등을 지정합니다.
    """
    model_name_or_path: str = field(
        default="sentence-transformers/all-MiniLM-L12-v2",
        metadata={"help": "Pretrained model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    데이터셋 관련 인자: 데이터 파일 경로와 샘플 수 제한 등을 지정합니다.
    """
    dataset_file: str = field(
        default=None, metadata={"help": "Path to the training dataset file (each line is a JSON with keys: source, positive, negative)"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging or quicker training, limit the number of training examples"}
    )

@dataclass
class CustomArguments:
    """
    추가적인 커스텀 인자: 예를 들어 Triplet loss의 margin 값을 지정합니다.
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
        # model_output[0]: last_hidden_state (batch, seq_len, hidden_size)
        token_embeddings = model_output[0]
        # attention_mask: (batch, seq_len) -> (batch, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

class SentenceTransformerModel(nn.Module):
    """
    기본 모델과 mean pooling layer를 결합한 SentenceTransformer 모델.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooling(outputs, attention_mask)
        return pooled


#############################################
#   Data Collator & Dataset
#############################################

class SentenceTransformerCollator:
    """
    (source, positive, negative) 데이터 예제를 토크나이즈하여 배치로 만드는 콜레이터.
    """
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        source_texts = [example["source"] for example in batch]
        positive_texts = [example["positive"] for example in batch]
        negative_texts = [example["negative"] for example in batch]

        source_enc = self.tokenizer(
            source_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        positive_enc = self.tokenizer(
            positive_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        negative_enc = self.tokenizer(
            negative_texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        return {"source": source_enc, "positive": positive_enc, "negative": negative_enc}

class SupervisedDataset(Dataset):
    """
    간단한 Dataset 클래스. 각 줄이 JSON 형식으로 {"source": ..., "positive": ..., "negative": ...} 이어야 합니다.
    """
    def __init__(self, file_path: str, max_samples: Optional[int] = None):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                if max_samples is not None and len(self.data) >= max_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#############################################
#   Custom Trainer
#############################################

class SentenceTransformerSupervisedTrainer(Trainer):
    """
    Trainer를 상속받아 (source, positive, negative) 데이터를 받아 TripletMarginLoss를 계산합니다.
    """
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs는 콜레이터에서 만든 딕셔너리입니다.
        source_enc = inputs["source"]
        positive_enc = inputs["positive"]
        negative_enc = inputs["negative"]

        # 모델이 있는 device로 옮기기
        device = model.base_model.device
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


#############################################
#   Main function
#############################################

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # 토크나이저와 base 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    base_model = AutoModel.from_pretrained(model_args.model_name_or_path)
    # 풀링 모듈을 포함하는 SentenceTransformer 모델 생성
    model = SentenceTransformerModel(base_model)
    logger.info("Loaded model and tokenizer from %s", model_args.model_name_or_path)

    # 데이터셋 로딩
    if data_args.dataset_file is None:
        raise ValueError("DataTrainingArguments에 dataset_file 경로를 지정해야 합니다.")
    train_dataset = SupervisedDataset(data_args.dataset_file, max_samples=data_args.max_train_samples)
    logger.info("Loaded %d training examples", len(train_dataset))

    # 데이터 콜레이터 생성 (max_length는 TrainingArguments의 max_seq_length 또는 기본값 128)
    max_length = training_args.max_seq_length if training_args.max_seq_length is not None else 128
    collator = SentenceTransformerCollator(tokenizer, max_length=max_length)

    # Trainer 생성
    trainer = SentenceTransformerSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        margin=custom_args.margin
    )

    # 학습 시작
    trainer.train()


if __name__ == "__main__":
    main()
