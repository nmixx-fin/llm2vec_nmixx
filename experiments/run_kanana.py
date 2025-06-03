import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
# Hugging Face Dataset과 이름 충돌 방지를 위해 TorchDataset으로 명명
from torch.utils.data import Dataset as TorchDataset 

import transformers
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    DataCollatorWithPadding, 
    set_seed,
)
from datasets import load_dataset, Dataset
import datasets

# modeling_kanana2vec.py와 configuration_kanana2vec.py가 동일 경로에 있다고 가정
from modeling_kanana2vec import Kanana2VecModel 
from configuration_kanana2vec import Kanana2VecConfig

from peft import LoraConfig, get_peft_model, PeftModel

# 로깅 기본 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)], 
)
logger = logging.getLogger(__name__)

########################################
# Argument Dataclasses
########################################

@dataclass
class ModelArguments:
    """모델 관련 인자"""
    model_name_or_path: str = field(
        default="kakaocorp/kanana-nano-2.1b-embedding",
        metadata={"help": "사전 학습된 모델의 Hugging Face Hub 경로 또는 로컬 경로"}
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "PEFT 어댑터 모델의 경로 (선택 사항, 추가 학습 또는 추론 시 사용)"}
    )
    torch_dtype: Optional[str] = field(
        default="auto", 
        metadata={
            "help": "모델 로드 시 사용할 torch.dtype. 'auto', 'float16', 'bfloat16', 'float32' 중 선택."
        },
    )
    trust_remote_code: bool = field(
        default=True, 
        metadata={"help": "모델 로딩 시 원격 코드 실행 허용 여부 (Kanana 모델 등에 필요할 수 있음)"}
    )

@dataclass
class DataTrainingArguments:
    """데이터셋 및 전처리 관련 인자"""
    dataset_name: str = field(
        default="nmixx-fin/NMIXX_train",
        metadata={"help": "사용할 데이터셋의 Hugging Face Hub 이름"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "데이터셋의 특정 설정 이름 (필요한 경우)"}
    )
    train_split_name: str = field(
        default="train", 
        metadata={"help": "학습에 사용할 데이터셋 스플릿 이름"}
    )
    source_column: str = field(
        default="source_text", 
        metadata={"help": "데이터셋에서 source 텍스트를 담고 있는 컬럼 이름"}
    )
    positive_column: str = field(
        default="positive", 
        metadata={"help": "데이터셋에서 positive 텍스트를 담고 있는 컬럼 이름"}
    )
    negative_column: str = field(
        default="negative", 
        metadata={"help": "데이터셋에서 negative 텍스트를 담고 있는 컬럼 이름"}
    )
    instruction_text: str = field(
        default="다음 문장과 의미적으로 가장 유사한 문장을 선택하세요:", 
        metadata={"help": "Source 텍스트 앞에 추가할 instruction. 비워두면 instruction 없이 처리."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "디버깅 또는 빠른 학습을 위해 사용할 최대 학습 샘플 수 제한"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "토큰화 시 최대 시퀀스 길이"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "데이터 전처리에 사용할 프로세스 수"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "캐시된 학습 및 평가 데이터셋 덮어쓰기 여부"}
    )
    trust_remote_code_dataset: bool = field( # 데이터셋 로드 시 trust_remote_code 추가
        default=True, 
        metadata={"help": "데이터셋 로딩 시 원격 코드 실행 허용 여부"}
    )


@dataclass
class CustomTrainingArguments:
    """Triplet Loss 및 PEFT (LoRA) 관련 커스텀 인자"""
    margin: float = field(
        default=1.0,
        metadata={"help": "TripletMarginLoss의 margin 값"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA의 r (rank) 파라미터. 0 이하면 LoRA를 적용하지 않음."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA의 alpha 파라미터."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA 레이어의 드롭아웃 확률."})
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        metadata={"help": "LoRA를 적용할 대상 모듈 이름 리스트. Llama 계열 기본값."}
    )

########################################
# Data Collator
########################################

class TripletDataCollator:
    """ Triplet 데이터를 토큰화하고 배치로 만드는 데이터 콜레이터 """
    def __init__(self, tokenizer, max_length: int, instruction: str, 
                 source_col: str, positive_col: str, negative_col: str, model_for_instruction_format=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction.strip() if instruction else ""
        self.source_col = source_col
        self.positive_col = positive_col
        self.negative_col = negative_col
        self.model_for_instruction_format = model_for_instruction_format # instruction 포맷팅을 위해 모델 전달
        
    def _prepare_text_with_instruction(self, text: str) -> str:
        """Instruction을 텍스트 앞에 결합합니다."""
        if self.instruction:
            if self.model_for_instruction_format and hasattr(self.model_for_instruction_format, 'format_instruction'):
                formatted_instruction = self.model_for_instruction_format.format_instruction(self.instruction)
                return f"{formatted_instruction} {text.strip()}"
            return f"{self.instruction} {text.strip()}" 
        return text.strip()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        source_texts = [self._prepare_text_with_instruction(example[self.source_col]) for example in features]
        positive_texts = [example[self.positive_col].strip() for example in features]
        negative_texts = [example[self.negative_col].strip() for example in features]
        
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

########################################
# Custom Trainer
########################################

class KananaTripletTrainer(Trainer):
    """ Triplet Loss를 사용하여 Kanana2VecModel을 학습하는 커스텀 Trainer """
    def __init__(self, margin: float, instruction_text: str = "", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.instruction_text_for_len = instruction_text.strip() if instruction_text else ""
        self.instruction_token_len = 0

        if self.model and hasattr(self.model, 'tokenizer') and self.model.tokenizer:
            self._calculate_instruction_token_len()
        else:
            # 모델이 PeftModel로 래핑된 경우, 원본 모델의 토크나이저 접근 필요
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'tokenizer') and self.model.base_model.tokenizer:
                 self._calculate_instruction_token_len(tokenizer=self.model.base_model.tokenizer, model_for_format=self.model.base_model.model)
            else:
                logger.warning("Model or tokenizer not fully available at Trainer init for instruction length calculation.")


    def _calculate_instruction_token_len(self, tokenizer=None, model_for_format=None):
        """ Instruction의 실제 토큰 길이를 계산합니다. """
        # PEFT 적용 시 self.model은 PeftModel일 수 있으므로, 실제 토크나이저/모델 접근 주의
        active_tokenizer = tokenizer if tokenizer else (self.model.tokenizer if hasattr(self.model, 'tokenizer') else None)
        active_model_for_format = model_for_format if model_for_format else \
                                  (self.model.base_model.model if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model') else \
                                  (self.model.model if hasattr(self.model, 'model') else self.model))


        if not active_tokenizer:
            logger.warning("Tokenizer not found for instruction length calculation.")
            self.instruction_token_len = 0
            return

        if self.instruction_text_for_len:
            try:
                formatted_instruction = self.instruction_text_for_len
                if hasattr(active_model_for_format, 'format_instruction'):
                    formatted_instruction = active_model_for_format.format_instruction(self.instruction_text_for_len)
                
                # add_special_tokens=False: format_instruction이 이미 BOS/EOS 등을 처리한다고 가정.
                # Kanana 모델의 format_instruction 구현에 따라 이 부분이 달라질 수 있음.
                # 일반적으로 instruction 자체의 순수 토큰 길이를 계산하는 것이 목적.
                self.instruction_token_len = len(active_tokenizer.encode(formatted_instruction, add_special_tokens=False))
                logger.info(f"Instruction for length calculation: '{formatted_instruction}', Calculated token length for instruction part: {self.instruction_token_len}")
            except Exception as e:
                logger.error(f"Error calculating instruction token length: {e}. Assuming 0.", exc_info=True)
                self.instruction_token_len = 0
        else:
            self.instruction_token_len = 0
            logger.info("No instruction text provided, instruction token length is 0.")


    def compute_loss(self, model: Union[torch.nn.Module, PeftModel], inputs: Dict[str, Dict[str, torch.Tensor]], return_outputs=False, **kwargs):
        logger.info(f"compute_loss: type(model) = {type(model)}")
        logger.info(f"compute_loss: type(self.model) = {type(self.model)}")
        if hasattr(model, 'module'):
            logger.info(f"compute_loss: type(model.module) = {type(model.module)}")
        if isinstance(self.model, PeftModel):
            logger.info(f"compute_loss: type(self.model.base_model) = {type(self.model.base_model)}")
            if hasattr(self.model.base_model, 'model'):
                logger.info(f"compute_loss: type(self.model.base_model.model) = {type(self.model.base_model.model)}")
        # 'model' 인자는 Trainer의 training_step에서 전달되며, DDP 등으로 래핑된 모델일 수 있음.
        # 실제 forward pass에는 이 'model' 인자를 사용해야 함.

        # Kanana2VecModel 고유의 헬퍼 메서드를 사용하기 위해 원본 Kanana2VecModel 인스턴스를 가져옴.
        
        # 1. DDP 래퍼 벗기기 (model 인자로부터)
        model_for_forward_unwrapped = model.module if hasattr(model, 'module') else model # forward pass에 사용될 기본 모델 (PeftModel or Kanana2VecModel)
        
        # 2. 헬퍼 메서드용 Kanana2VecModel 인스턴스 가져오기
        # self.model은 Trainer 초기화 시 전달된 모델 (PeftModel(Kanana2VecModel) 또는 Kanana2VecModel)
        temp_model_for_helpers = self.model # Trainer가 __init__에서 받은 모델
        if hasattr(temp_model_for_helpers, 'module'): # 만약 self.model도 DDP 래핑되어 있다면 (일반적이지 않음)
            temp_model_for_helpers = temp_model_for_helpers.module

        if isinstance(temp_model_for_helpers, PeftModel):
            # PeftModel의 base_model은 LoraModel 등을 반환하고, 그 내부의 model이 원본 Kanana2VecModel
            if hasattr(temp_model_for_helpers.base_model, 'model') and \
               isinstance(temp_model_for_helpers.base_model.model, Kanana2VecModel):
                kanana_model_instance = temp_model_for_helpers.base_model.model
            # 또는 PeftModel이 Kanana2VecModel을 직접 감싼 경우 (base_model이 Kanana2VecModel)
            elif isinstance(temp_model_for_helpers.base_model, Kanana2VecModel):
                 kanana_model_instance = temp_model_for_helpers.base_model
            else:
                raise TypeError(f"PEFT base model (type: {type(temp_model_for_helpers.base_model)}) or its inner model is not Kanana2VecModel.")
        elif isinstance(temp_model_for_helpers, Kanana2VecModel):
            kanana_model_instance = temp_model_for_helpers # PEFT가 적용되지 않은 경우
        else:
            raise TypeError(f"self.model (type: {type(temp_model_for_helpers)}) is not PeftModel or Kanana2VecModel.")

        # --- 필수 속성/메서드 검사 ---
        # Kanana2VecModel 인스턴스가 맞는지 한번 더 확인 (필수 메서드 존재 여부로)
        # 여기서 kanana_model_instance는 순수 Kanana2VecModel 이어야 함.
        if not hasattr(kanana_model_instance, 'tokenizer'):
            # main에서 model.tokenizer = tokenizer 할당이 PEFT 적용 전에 이루어짐.
            # PEFT 적용 후에는 peft_model.base_model.model.tokenizer 로 접근해야 할 수 있음.
            # 또는, Trainer가 초기화될 때 tokenizer를 별도로 가지고 있으므로 self.tokenizer 사용도 가능.
            # 여기서는 kanana_model_instance에 tokenizer가 이미 할당되어 있다고 가정.
            # 만약 여기서 None이라면, _calculate_instruction_token_len 내부에서 tokenizer를 self.tokenizer로 대체해야 함.
             logger.warning(f"kanana_model_instance (type: {type(kanana_model_instance)}) does not have 'tokenizer' attribute. "
                           "Instruction length calculation might use Trainer's tokenizer.")
        
        if not (hasattr(kanana_model_instance, 'prepare_kwargs_from_batch') and \
                hasattr(kanana_model_instance, 'format_instruction')):
            # 위에서 tokenizer 검사는 별도로 하고, 여기서는 핵심 메서드만 확인
            raise TypeError(f"The obtained model instance for helpers (type: {type(kanana_model_instance)}) "
                            "does not have 'prepare_kwargs_from_batch' or 'format_instruction'.")
        # --- 검사 끝 ---

        # instruction 토큰 길이 계산 (필요시)
        if self.instruction_token_len == 0 and self.instruction_text_for_len:
             self._calculate_instruction_token_len(
                 tokenizer=(kanana_model_instance.tokenizer if hasattr(kanana_model_instance, 'tokenizer') and kanana_model_instance.tokenizer else self.tokenizer), # 안전하게 접근
                 model_for_format=kanana_model_instance
             )
        
        device = next(model.parameters()).device # forward pass용 모델의 device 사용

        source_batch = {k: v.to(device) for k, v in inputs["source"].items()}
        positive_batch = {k: v.to(device) for k, v in inputs["positive"].items()}
        negative_batch = {k: v.to(device) for k, v in inputs["negative"].items()}
        
        source_features = kanana_model_instance.prepare_kwargs_from_batch(source_batch, instruction_lens=self.instruction_token_len, device=device)
        positive_features = kanana_model_instance.prepare_kwargs_from_batch(positive_batch, instruction_lens=0, device=device) 
        negative_features = kanana_model_instance.prepare_kwargs_from_batch(negative_batch, instruction_lens=0, device=device)

        # 실제 forward pass는 Trainer가 전달한 'model' (DDP/PEFT 래핑된)을 사용
        source_emb_output = model(**source_features)
        positive_emb_output = model(**positive_features)
        negative_emb_output = model(**negative_features)

        if not (hasattr(source_emb_output, 'embedding') and hasattr(positive_emb_output, 'embedding') and hasattr(negative_emb_output, 'embedding')):
            raise AttributeError("The model's forward pass did not return an output object with an 'embedding' attribute.")

        source_emb = source_emb_output.embedding
        positive_emb = positive_emb_output.embedding
        negative_emb = negative_emb_output.embedding

        loss = self.triplet_loss(source_emb, positive_emb, negative_emb)
        
        if return_outputs:
            return loss, {"source_emb": source_emb, "positive_emb": positive_emb, "negative_emb": negative_emb}
        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Attempting to save model to {output_dir}")

        # self.model은 PEFT가 적용된 PeftModel 인스턴스일 수 있음
        model_to_save = self.model_wrapped if hasattr(self, 'model_wrapped') else self.model
        tokenizer_to_save = self.tokenizer
        
        if isinstance(model_to_save, PeftModel):
            try:
                logger.info("Model is a PeftModel. Merging PEFT adapter into the base model...")
                # merge_and_unload()는 PeftModel의 메서드
                model_to_save = model_to_save.merge_and_unload() 
                # 병합 후에는 기본 모델(Kanana2VecModel)의 tokenizer를 사용해야 함
                if hasattr(model_to_save, 'tokenizer') and model_to_save.tokenizer:
                    tokenizer_to_save = model_to_save.tokenizer
                logger.info("PEFT adapter merged successfully. Base model will be saved.")
            except Exception as e:
                logger.error(f"Failed to merge PEFT adapter: {e}. Saving PEFT adapter separately.", exc_info=True)
                self.model.save_pretrained(os.path.join(output_dir, "peft_adapter_on_merge_fail"))
                if tokenizer_to_save: # PEFT 모델의 토크나이저 (보통 base와 동일)
                    tokenizer_to_save.save_pretrained(output_dir)
                if self.args:
                     torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                return
        else:
            logger.info("Model is not a PeftModel instance, saving as is.")
            if hasattr(model_to_save, 'tokenizer') and model_to_save.tokenizer: # Kanana2VecModel의 토크나이저
                 tokenizer_to_save = model_to_save.tokenizer


        if hasattr(model_to_save, 'save_pretrained'):
             model_to_save.save_pretrained(output_dir)
             logger.info(f"Full model (merged if applicable) saved to {output_dir}")
        else:
             logger.warning("Model object does not have a 'save_pretrained' method. Defaulting to Trainer's _save.")
             super()._save(output_dir, _internal_call=_internal_call) 

        if tokenizer_to_save:
            tokenizer_to_save.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
        else:
            logger.warning("Could not determine tokenizer to save.")

        if self.args:
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            logger.info(f"Training arguments saved to {output_dir}")


########################################
# Main Training Routine
########################################

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        logger.info(f"Loading arguments from JSON config file: {sys.argv[1]}")
        # parse_json_file은 튜플을 반환하므로, 각 변수에 맞게 할당
        args_tuple = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, data_args, training_args, custom_args = args_tuple
    else:
        logger.info("Loading arguments from command line.")
        # parse_args_into_dataclasses도 튜플을 반환
        args_tuple = parser.parse_args_into_dataclasses()
        model_args, data_args, training_args, custom_args = args_tuple


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
                f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits ({training_args.fp16_backend}): {training_args.fp16 or training_args.bf16}")
    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(f"Model arguments:\n{model_args}")
    logger.info(f"Data arguments:\n{data_args}")
    logger.info(f"Custom training arguments:\n{custom_args}")

    set_seed(training_args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=model_args.trust_remote_code
        )
        
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}
        torch_dtype_value = dtype_map.get(str(model_args.torch_dtype).lower(), "auto")
        
        logger.info(f"Attempting to load model '{model_args.model_name_or_path}' with torch_dtype='{torch_dtype_value}'.")

        model = Kanana2VecModel.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype_value,
        )
        
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            model.tokenizer = tokenizer
            logger.info("Manually assigned externally loaded tokenizer to the model instance.")
        elif model.tokenizer.name_or_path != tokenizer.name_or_path:
             logger.warning(f"Model's internal tokenizer ({model.tokenizer.name_or_path}) "
                           f"differs from the one loaded separately ({tokenizer.name_or_path}). "
                           "Ensure consistency if this is not intended.")
        
        logger.info(f"Successfully loaded model and tokenizer.")

    except Exception as e:
        logger.error(f"Error loading base model or tokenizer from '{model_args.model_name_or_path}': {e}", exc_info=True)
        sys.exit(1)

    if custom_args.lora_r > 0:
        logger.info(f"Applying LoRA: r={custom_args.lora_r}, alpha={custom_args.lora_alpha}, modules={custom_args.lora_target_modules}")
        # Kanana2VecModel의 base_model (즉, BiLlamaModel)에 LoRA를 적용해야 할 수 있음.
        # get_peft_model은 PreTrainedModel을 받으므로, Kanana2VecModel 자체에 적용 가능.
        peft_config = LoraConfig(
            r=custom_args.lora_r,
            lora_alpha=custom_args.lora_alpha,
            lora_dropout=custom_args.lora_dropout,
            target_modules=custom_args.lora_target_modules,
            bias="none", 
            task_type=None, 
        )
        try:
            model = get_peft_model(model, peft_config) # model은 이제 PeftModel 타입이 됨
            logger.info("Successfully applied LoRA to the model.")
            model.print_trainable_parameters()
            
            if model_args.peft_model_name_or_path:
                logger.info(f"Loading existing PEFT adapter weights from {model_args.peft_model_name_or_path}")
                model.load_adapter(model_args.peft_model_name_or_path, "default") 
                logger.info(f"Successfully loaded PEFT adapter weights.")

        except Exception as e:
            logger.error(f"Error applying PEFT (LoRA): {e}. Training with the base model (if LoRA was optional).", exc_info=True)
            # LoRA 적용이 필수라면 여기서 종료, 선택 사항이면 경고 후 진행
            # sys.exit(1) # LoRA 적용이 필수인 경우
    else:
        logger.info("LoRA is not applied (lora_r <= 0).")

    try:
        logger.info(f"Loading dataset '{data_args.dataset_name}' (config: '{data_args.dataset_config_name or 'default'}', split: '{data_args.train_split_name}').")
        raw_dataset = load_dataset(
            data_args.dataset_name, 
            name=data_args.dataset_config_name,
            split=data_args.train_split_name,
            trust_remote_code=data_args.trust_remote_code_dataset 
        )
        logger.info(f"Dataset loaded. Number of examples: {len(raw_dataset)}")

        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            logger.info(f"Selecting a subset of {data_args.max_train_samples} training samples.")
            raw_dataset = raw_dataset.select(range(min(data_args.max_train_samples, len(raw_dataset))))
            logger.info(f"Dataset reduced to {len(raw_dataset)} samples.")

        required_cols = [data_args.source_column, data_args.positive_column, data_args.negative_column]
        missing_cols = [col for col in required_cols if col not in raw_dataset.column_names]
        if missing_cols:
            logger.error(f"Missing required columns in the dataset: {missing_cols}. Available columns: {raw_dataset.column_names}")
            sys.exit(1)
        
        train_dataset = raw_dataset

    except Exception as e:
        logger.error(f"Error loading or processing dataset '{data_args.dataset_name}': {e}", exc_info=True)
        sys.exit(1)
    
    model_instance_for_collator = model.base_model if isinstance(model, PeftModel) else model

    data_collator = TripletDataCollator(
        tokenizer=tokenizer, # 외부에서 로드한 토크나이저 사용 일관성 유지
        max_length=data_args.max_seq_length,
        instruction=data_args.instruction_text,
        source_col=data_args.source_column,
        positive_col=data_args.positive_column,
        negative_col=data_args.negative_column,
        model_for_instruction_format=model_instance_for_collator # 모델 인스턴스 전달
    )

    trainer = KananaTripletTrainer(
        model=model, # PEFT 적용된 모델 전달
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer, 
        margin=custom_args.margin,
        instruction_text=data_args.instruction_text,
    )

    try:
        logger.info("Starting model training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training finished.")

        final_save_path = os.path.join(training_args.output_dir, "final_model_merged")
        logger.info(f"Saving final model to: {final_save_path}")
        trainer.save_model(final_save_path) 
        
        metrics = train_result.metrics
        max_train_samples_actual = data_args.max_train_samples if data_args.max_train_samples is not None and data_args.max_train_samples > 0 else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples_actual, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current model state (if applicable)...")
        if training_args.should_save: # save_strategy가 "no"가 아닐 때만 저장 시도
            interrupted_save_path = os.path.join(training_args.output_dir, "interrupted_checkpoint")
            trainer.save_model(interrupted_save_path)
            logger.info(f"Model state (potentially adapter only if merge failed) saved to {interrupted_save_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()