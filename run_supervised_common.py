import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers.trainer_utils import seed_worker
from datasets import load_dataset
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
    TrainerCallback,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from accelerate.logging import get_logger
from llm2vec import LLM2Vec
from llm2vec.loss.utils import load_loss

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use from Huggingface Hub."},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )


@dataclass
class DataSample:
    id_: int
    query: str
    positive: str
    negative: str = None


class TrainSample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0
    ):
        """
        Creates one TrainSample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 호출되는 메소드"""
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True
        return control


class BaseSupervisedTrainer:
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Override the default training_step to handle our specific input format.
        """
        model.train()

        # 직접 compute_loss를 호출하여 labels가 모델에 전달되는 것을 방지
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 원래 Trainer의 구현과 유사하게 backward 호출
        self.accelerator.backward(loss)

        return loss.detach()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs

        # 입력을 모델의 디바이스로 이동
        device = next(model.parameters()).device
        features = [f.to(device) for f in features]

        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        # 모든 텐서가 같은 디바이스에 있는지 확인
        if d_reps_neg is not None:
            d_reps_neg = d_reps_neg.to(device)
        q_reps = q_reps.to(device)
        d_reps = d_reps.to(device)

        loss = self.loss_function(q_reps, d_reps, d_reps_neg)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def get_train_dataloader(self) -> DataLoader:
        # Copying most of the code from the parent class, changing the sampler to SequentialSampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)


def load_custom_dataset(dataset_name=None, file_path=None, split="train", **kwargs):
    """
    Loads a custom dataset from Huggingface Hub or local file.

    Args:
        dataset_name (str): Name of the dataset to load from Huggingface Hub.
        file_path (str): Path to the dataset file.
        split (str): Split of the dataset to load.
    """
    if dataset_name:
        # 허깅페이스 데이터셋 로드
        dataset = load_dataset(dataset_name, split=split)
    elif file_path:
        # 로컬 파일에서 데이터셋 로드
        dataset = load_dataset("json", data_files=file_path, split=split)
    else:
        raise ValueError("Either dataset_name or file_path must be provided.")

    # 데이터셋에 필요한 컬럼이 있는지 확인
    required_columns = ["query", "positive", "negative"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must contain column '{col}'")

    # DataSample 객체로 변환
    samples = []
    for i, item in enumerate(dataset):
        samples.append(
            DataSample(
                id_=i,
                query=item["query"],
                positive=item["positive"],
                negative=item["negative"],
            )
        )

    # TrainSample 객체로 변환
    train_samples = []
    for sample in samples:
        train_samples.append(
            TrainSample(
                guid=str(sample.id_),
                texts=[sample.query, sample.positive, sample.negative],
                label=1.0,
            )
        )

    return train_samples


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    """
    모델에 따라 입력 텍스트를 적절히 가공합니다.
    """
    # BGE 모델
    if model.config._name_or_path == "BAAI/bge-large-en-v1.5":
        text = (
            "Represent this sentence for searching relevant passages: " + text.strip()
        )
    # Instructor 모델
    elif model.config._name_or_path == "hkunlp/instructor-base":
        text = "Represent the following text for retrieval: " + text.strip()
    # LLama 3 모델
    elif model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
    # Mistral과 Llama 2 모델
    elif model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    # Gemma 모델
    elif model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    # Qwen 모델
    elif model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"

    # pooling_mode가 eos_token인 경우 모델별로 EOS 토큰 추가
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        if model.config.__class__.__name__ == "T5Config":
            lora_modules = [
                "q",  # T5 Self-Attention query
                "k",  # T5 Self-Attention key
                "v",  # T5 Self-Attention value
                "o",  # T5 Self-Attention output
                "wi",  # T5 Feed Forward input
                "wo",  # T5 Feed Forward output
            ]
        else:  # BGE-Large 등 다른 모델용
            lora_modules = [
                "query",
                "key",
                "value",
                "dense",
            ]

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            # T5 모델을 위한 decoder_input_ids 추가 (instructor 모델용)
            if (
                hasattr(self.model.model, "config")
                and hasattr(self.model.model.config, "_name_or_path")
                and (
                    "instructor" in self.model.model.config._name_or_path.lower()
                    or hasattr(self.model.model.config, "is_encoder_decoder")
                    and self.model.model.config.is_encoder_decoder
                )
            ):
                # Shift input_ids 오른쪽으로 한 칸 이동 (T5 디코더 입력 생성 방식)
                decoder_input_ids = tokenized["input_ids"].clone()
                decoder_input_ids = torch.nn.functional.pad(
                    decoder_input_ids[:, :-1], (1, 0), value=0
                )
                tokenized["decoder_input_ids"] = decoder_input_ids
            sentence_features.append(tokenized)

        return sentence_features, labels


class LLM2VecSupervisedTrainer(BaseSupervisedTrainer, Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # LoRA 어댑터를 원본 모델에 병합
        if hasattr(self.model.model, "merge_and_unload"):
            merged_model = self.model.model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            # 기존 저장 방식
            self.model.model.save_pretrained(output_dir)

        # pooling_mode 설정을 저장
        pooling_mode_path = os.path.join(output_dir, "pooling_mode.txt")
        with open(pooling_mode_path, "w") as f:
            f.write(self.model.pooling_mode)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
