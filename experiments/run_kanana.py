#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)

# Kanana2VecModel이 modeling_kanana2vec.py 파일 내에 있다고 가정합니다.
from modeling_kanana2vec import Kanana2VecModel

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


########################################
# Argument Dataclasses
########################################

@dataclass
class ModelArguments:
    """
    모델 관련 인자: pretrained 모델 경로 등을 지정합니다.
    """
    model_name_or_path: str = field(
        default="kakaocorp/kanana-nano-2.1b-embedding",
        metadata={"help": "Pretrained model identifier from huggingface.co"}
    )

@dataclass
class DataTrainingArguments:
    """
    데이터셋 관련 인자: 데이터 파일 경로와 샘플 수 제한, 최대 시퀀스 길이 등을 지정합니다.
    """
    dataset_file: str = field(
        default=None,
        metadata={"help": "Path to the training dataset file. Each line must be a JSON with keys: source, positive, negative"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, limit the number of training examples"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization"}
    )

@dataclass
class CustomArguments:
    """
    추가 커스텀 인자: 예를 들어 TripletLoss의 margin 값을 지정합니다.
    """
    margin: float = field(
        default=1.0,
        metadata={"help": "Margin value for TripletMarginLoss"}
    )


########################################
# Dataset & Data Collator
########################################

class TripletDataset(Dataset):
    """
    각 줄마다 JSON 객체로 {"source": ..., "positive": ..., "negative": ...} 형태여야 하는 데이터셋.
    """
    def __init__(self, file_path: str, max_samples: Optional[int] = None):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                if "source" in example and "positive" in example and "negative" in example:
                    self.data.append(example)
                else:
                    logger.warning("Skipping example without required keys: %s", line)
                if max_samples is not None and len(self.data) >= max_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Kanana2VecCollator:
    """
    각 예제의 source, positive, negative 텍스트를 토크나이저를 통해 인코딩하여 배치로 만드는 콜레이터.
    """
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        source_texts = [ex["source"] for ex in batch]
        positive_texts = [ex["positive"] for ex in batch]
        negative_texts = [ex["negative"] for ex in batch]

        source_enc = self.tokenizer(
            source_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive_enc = self.tokenizer(
            positive_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        negative_enc = self.tokenizer(
            negative_texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"source": source_enc, "positive": positive_enc, "negative": negative_enc}


########################################
# Custom Trainer
########################################

class Kanana2VecSupervisedTrainer(Trainer):
    """
    Kanana2VecModel을 대상으로 (source, positive, negative) 데이터를 받아 TripletMarginLoss를 계산하는 Trainer.
    """
    def __init__(self, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def compute_loss(self, model: torch.nn.Module, inputs, return_outputs=False):
        device = next(model.parameters()).device

        # 각 branch에 대해 토크나이즈된 딕셔너리를 추출합니다.
        source_batch = inputs["source"]
        positive_batch = inputs["positive"]
        negative_batch = inputs["negative"]

        # Kanana2VecModel은 prepare_kwargs_from_batch() 메서드를 통해
        # input_ids, attention_mask, pool_mask 등을 준비합니다.
        # instruction_lens는 여기서는 사용하지 않으므로 0으로 전달합니다.
        source_features = model.prepare_kwargs_from_batch(source_batch, instruction_lens=0, device=device)
        positive_features = model.prepare_kwargs_from_batch(positive_batch, instruction_lens=0, device=device)
        negative_features = model.prepare_kwargs_from_batch(negative_batch, instruction_lens=0, device=device)

        # 각 branch의 embedding 계산 (model.forward()는 EmbeddingModelOutput을 반환)
        source_emb = model(**source_features).embedding
        positive_emb = model(**positive_features).embedding
        negative_emb = model(**negative_features).embedding

        loss = self.triplet_loss(source_emb, positive_emb, negative_emb)
        if return_outputs:
            return loss, {"source_emb": source_emb, "positive_emb": positive_emb, "negative_emb": negative_emb}
        return loss


########################################
# Main Training Routine
########################################

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # 모델과 토크나이저 로딩 (trust_remote_code 옵션 사용)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = Kanana2VecModel.from_pretrained(model_args.model_name_or_path)
    logger.info("Loaded Kanana2VecModel from %s", model_args.model_name_or_path)

    # 학습 데이터셋 로딩
    if not data_args.dataset_file:
        raise ValueError("DataTrainingArguments에 dataset_file 경로를 지정해야 합니다.")
    train_dataset = TripletDataset(data_args.dataset_file, max_samples=data_args.max_train_samples)
    logger.info("Loaded %d training examples", len(train_dataset))

    # 데이터 콜레이터 생성
    collator = Kanana2VecCollator(tokenizer, max_length=data_args.max_seq_length)

    # Trainer 생성
    trainer = Kanana2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        margin=custom_args.margin,
    )

    # 학습 시작
    trainer.train()


if __name__ == "__main__":
    main()
