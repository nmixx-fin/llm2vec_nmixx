from dataclasses import dataclass
from typing import List, Union, Dict, Mapping, TypedDict, Optional, Tuple
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import os
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaModel,
)
from transformers.utils import (
    logging,
    ModelOutput,
)
from configuration_kanana2vec import Kanana2VecConfig

logger = logging.get_logger(__name__)

def _move_to_device(maybe_tensor, device: torch.device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device, non_blocking=device.type == "cuda")
    elif isinstance(maybe_tensor, dict):
        return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
    elif isinstance(maybe_tensor, list):
        return [_move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([_move_to_device(x, device) for x in maybe_tensor])
    elif isinstance(maybe_tensor, Mapping):
        return type(maybe_tensor)({k: _move_to_device(v, device) for k, v in maybe_tensor.items()})
    else:
        return maybe_tensor

def move_to_device(sample, device: torch.device):
    if device.type == "cpu":
        return sample
    
    if len(sample) == 0:
        return {}
    return _move_to_device(sample, device)

def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if len(instruction) > 0:
        examples['text'] = [f"{instruction.strip()} {text.strip()}" for text in examples['text']]
    else:
        examples['text'] = [f"{text.strip()}" for text in examples['text']]
        
    batch_dict = tokenizer(
        examples['text'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True)
    return batch_dict

def format_instruction(instruction: str):
    return f"Instruct: {instruction}\nQuery:" if len(instruction) > 0 else ""


class Kanana2VecFeatures(TypedDict):
    input_dict: torch.Tensor
    attention_mask: torch.Tensor
    pool_mask: torch.Tensor

@dataclass
class EmbeddingModelOutput(ModelOutput):
    embedding: torch.FloatTensor = None


class BiLlamaModel(LlamaModel):
    config_class = Kanana2VecConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    # @staticmethod
    # def _prepare_4d_causal_attention_mask_with_cache_position(
    #     attention_mask: torch.Tensor,
    #     sequence_length: int,
    #     target_length: int,
    #     dtype: torch.dtype,
    #     device: torch.device,
    #     cache_position: torch.Tensor,
    #     batch_size: int,
    #     **kwargs,
    # ):
    #     """
    #     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    #     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    #     Args:
    #         attention_mask (`torch.Tensor`):
    #             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
    #             `(batch_size, 1, query_length, key_value_length)`.
    #         sequence_length (`int`):
    #             The sequence length being processed.
    #         target_length (`int`):
    #             The target length: when generating with static cache, the mask should be as long as the static cache,
    #             to account for the 0 padding, the part of the cache that is not filled yet.
    #         dtype (`torch.dtype`):
    #             The dtype to use for the 4D attention mask.
    #         device (`torch.device`):
    #             The device to plcae the 4D attention mask on.
    #         cache_position (`torch.Tensor`):
    #             Indices depicting the position of the input sequence tokens in the sequence.
    #         batch_size (`torch.Tensor`):
    #             Batch size.
    #     """
    #     if attention_mask is not None and attention_mask.dim() == 4:
    #         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    #         causal_mask = attention_mask
    #     else:
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = torch.zeros(
    #             (sequence_length, target_length), dtype=dtype, device=device
    #         )
    #         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    #         if attention_mask is not None:
    #             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
    #             padding_mask = padding_mask == 0
    #             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #                 padding_mask, min_dtype
    #             )
    #     return causal_mask

class Kanana2VecModel(PreTrainedModel):
    config_class = Kanana2VecConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: Kanana2VecConfig):
        super().__init__(config)
        self.model = BiLlamaModel(config) 
        self.tokenizer = None 
        # self.add_pad_token() # 토크나이저 설정 후 호출

    def add_pad_token(self):
        if self.tokenizer and self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                logger.warning("eos_token is not set, could not automatically set pad_token.")

    # 이 메서드가 클래스 내에 정의되어 있어야 합니다.
    def format_instruction(self, instruction: str) -> str:
        """
        모델에 특화된 방식으로 instruction 문자열을 포맷팅합니다.
        modeling_kanana2vec.py 파일 최상단에 정의된 전역 format_instruction 함수를 사용합니다.
        """
        # 전역 format_instruction 함수가 이 파일 내에 정의되어 있다고 가정합니다.
        # 만약 없다면, 아래 주석처럼 직접 구현하거나 전역 함수를 임포트해야 합니다.
        # 예: return f"Instruct: {instruction.strip()}\nQuery:" if instruction and instruction.strip() else ""
        return format_instruction(instruction) # 파일 내 전역 함수 호출

    # 이 메서드가 클래스 내에 정의되어 있어야 합니다.
    def prepare_kwargs_from_batch(self, batch_dict: Dict[str, torch.Tensor], instruction_lens: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        토큰화된 배치와 instruction 길이를 받아 모델 forward에 필요한 kwargs를 준비합니다.
        Kanana2VecFeatures 타입 힌트 대신 Dict를 사용합니다.
        """
        moved_batch = move_to_device(batch_dict, device)

        input_ids = moved_batch['input_ids']
        attention_mask_original = moved_batch['attention_mask']

        pool_mask = attention_mask_original.clone()
        if instruction_lens > 0:
            for i in range(pool_mask.size(0)):
                actual_seq_len = attention_mask_original[i].sum().item()
                # instruction_lens가 실제 시퀀스 길이를 초과하지 않도록 min 사용
                mask_len = min(instruction_lens, actual_seq_len) 
                pool_mask[i, :mask_len] = 0 # instruction 부분은 풀링에서 제외

        features: Dict[str, torch.Tensor] = { # Kanana2VecFeatures 대신 Dict 사용
            'input_ids': input_ids,
            'attention_mask': attention_mask_original,
            'pool_mask': pool_mask,
        }
        return features

    # ... (get_input_embeddings, set_input_embeddings, forward, from_pretrained, save_pretrained 등 나머지 메서드들) ...
    # forward 메서드 내에서 self.model 호출 시, BiLlamaModel의 forward가 호출됨.
    # BiLlamaModel의 __init__에서 self.layers[i].self_attn.is_causal = False 설정이 중요.
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pool_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs, 
        ) -> Union[Tuple, EmbeddingModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model( # self.model은 BiLlamaModel 인스턴스
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, 
        )
        last_hidden_states = model_outputs.last_hidden_state

        if pool_mask is None:
            logger.warning("pool_mask not provided to forward pass. Using attention_mask for pooling, which includes instruction tokens.")
            pool_mask_to_use = attention_mask if attention_mask is not None else torch.ones_like(input_ids, device=last_hidden_states.device)
        else:
            pool_mask_to_use = pool_mask.to(last_hidden_states.device)

        expanded_pool_mask = pool_mask_to_use.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_states * expanded_pool_mask, dim=1)
        sum_mask = torch.clamp(expanded_pool_mask.sum(dim=1), min=1e-9) 

        embedding = sum_embeddings / sum_mask

        if not return_dict:
            return (embedding,)

        return EmbeddingModelOutput(embedding=embedding)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        trust_remote_code_flag = kwargs.get('trust_remote_code', True)
        config_load_kwargs = {k: v for k, v in kwargs.items() if k in ['revision', 'token', 'cache_dir']}

        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code_flag,
                **config_load_kwargs
            )
            if not isinstance(config, Kanana2VecConfig):
                logger.info(f"Loaded config is {type(config)}. Adapting to Kanana2VecConfig.")
                kanana_config_dict = config.to_dict()
                config = Kanana2VecConfig(**kanana_config_dict) # Kanana2VecConfig의 필드와 일치해야 함
        except Exception as e:
            logger.error(f"Error loading or adapting config for {pretrained_model_name_or_path}: {e}", exc_info=True)
            raise

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=trust_remote_code_flag,
                **config_load_kwargs
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer for {pretrained_model_name_or_path}: {e}", exc_info=True)
            raise

        # config 인스턴스를 전달하여 PreTrainedModel.from_pretrained 호출
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        model.tokenizer = tokenizer
        model.add_pad_token()

        logger.info(f"Kanana2VecModel '{pretrained_model_name_or_path}' loaded with tokenizer.")
        return model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(save_directory)
                logger.info(f"Tokenizer also saved to {save_directory}")
            except Exception as e:
                logger.error(f"Error saving tokenizer to {save_directory}: {e}", exc_info=True)
        else:
            logger.warning(f"Tokenizer not found in the model instance, not saving tokenizer to {save_directory}")
