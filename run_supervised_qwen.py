import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from llm2vec import LLM2Vec
from llm2vec.loss.utils import load_loss
from llm2vec.experiment_utils import generate_experiment_id

from run_supervised_common import (
    ModelArguments,
    DataTrainingArguments,
    CustomArguments,
    DataSample,
    TrainSample,
    StopTrainingCallback,
    BaseSupervisedTrainer,
    LLM2VecSupervisedTrainer,
    load_custom_dataset,
    initialize_peft,
    DefaultCollator,
)

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if custom_args.experiment_id is not None:
        experiment_id = custom_args.experiment_id
    else:
        experiment_id = generate_experiment_id(
            name=data_args.dataset_name or "custom_dataset",
            split="train",
            model_name=(
                model_args.model_name_or_path
                if "/" not in model_args.model_name_or_path
                else model_args.model_name_or_path.split("/")[-1]
            ),
            pooling_mode=model_args.pooling_mode,
            train_batch_size=training_args.per_device_train_batch_size
            * accelerator.num_processes
            * training_args.gradient_accumulation_steps,
            max_seq_length=model_args.max_seq_length,
            bidirectional=model_args.bidirectional,
            epochs=training_args.num_train_epochs,
            seed=training_args.seed,
            warmup_steps=training_args.warmup_steps,
            lr=training_args.learning_rate,
            lora_r=custom_args.lora_r,
        )

    training_args.output_dir = f"{training_args.output_dir}/{experiment_id}"

    # 커스텀 데이터셋 로드
    train_examples = load_custom_dataset(
        dataset_name=data_args.dataset_name,
        file_path=data_args.dataset_file_path,
        split="train",
        effective_batch_size=training_args.per_device_train_batch_size
        * accelerator.num_processes,
    )

    # 최대 샘플 수 제한 (디버깅용)
    if data_args.max_train_samples is not None:
        train_examples = train_examples[: data_args.max_train_samples]

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    # Alibaba-NLP/gte-Qwen2-1.5B-instruct 모델 로드
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path
        or "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer

    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    trainer = LLM2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()


if __name__ == "__main__":
    main()
