from dataclasses import dataclass
from typing import List, Union, Dict, Mapping, TypedDict
from functools import partial
from tqdm.auto import tqdm
import numpy as np

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaModel,
    LLAMA_INPUTS_DOCSTRING,
)
from transformers.utils import (
    logging,
    ModelOutput,
)
from .configuration_kanana2vec import Kanana2VecConfig


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

def format_insruction(instruction: str):
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

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

class Kanana2VecModel(PreTrainedModel):
    config_class = Kanana2VecConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(self, config: Kanana2VecConfig):
        super().__init__(config)
        self.model = BiLlamaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, trust_remote_code=True)
        self.add_pad_token()

    def add_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_kwargs_from_batch(self, batch_dict: dict, instruction_lens: int, device: torch.device):
        batch_dict = move_to_device(batch_dict, device)
        attention_mask = batch_dict['attention_mask'].clone()
        attention_mask[:, :instruction_lens] = 0
        features: Kanana2VecFeatures = {
            'input_ids': batch_dict['input_ids'],
            'attention_mask': batch_dict['attention_mask'],
            'pool_mask': attention_mask,
        }
        return features

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pool_mask: torch.Tensor,
            return_dict: bool=True,
            **kwargs,
        ):
        last_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pool_mask = pool_mask.to(last_hidden_states.device)
        s = torch.sum(last_hidden_states * pool_mask.unsqueeze(-1).float(), dim=1)
        d = pool_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        if not return_dict:
            return (embedding,)
        return EmbeddingModelOutput(embedding=embedding)

    @torch.no_grad()
    def _do_encode(self,
        sentences: List[str],
        batch_size: int = 1,
        instruction: str = "",
        max_length: int = 512,
        num_workers: int = 0,
        **kwargs
    ) -> Union[np.ndarray, torch.FloatTensor]:
        dataset: Dataset = Dataset.from_dict({'text': sentences})
        instruction = format_insruction(instruction)
        dataset.set_transform(partial(input_transform_func,
                                      self.tokenizer,
                                      max_length=max_length,
                                      instruction=instruction))

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = num_workers,
            collate_fn = data_collator,
            pin_memory = True,
        )
        instruction_lens = len(self.tokenizer.encode(instruction)) if len(instruction) > 0 else 0

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc='encoding', mininterval=10):
            features = self.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=self.device)
            embeds=self(**features).embedding
            encoded_embeds.append(embeds)
        encoded_embeds = torch.cat(encoded_embeds, axis=0)
        if "return_numpy" in kwargs and  kwargs.get("return_numpy"):
            encoded_embeds = encoded_embeds.cpu().detach().numpy()
        return encoded_embeds

    @torch.no_grad()
    def encode(self, sentences: List[str], instruction: str="", max_length: int=512, **kwargs):
        instruction = format_insruction(instruction)
        instruction_lens = len(self.tokenizer.encode(instruction)) if len(instruction) > 0 else 0

        batch_dict = input_transform_func(
            self.tokenizer,
            {'text': sentences},
            max_length=max_length,
            instruction=instruction,
        )
        features: Kanana2VecFeatures = self.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=self.device)
        return self.forward(**features).embedding