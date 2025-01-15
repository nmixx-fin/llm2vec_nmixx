import torch
from transformers import AutoTokenizer, LlamaConfig
from llm2vec.models.bidirectional_pooling_llama import LlamaBiPoolingModel

# 모델과 토크나이저 초기화
model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = LlamaConfig.from_pretrained(model_name)
config.return_dict = False

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
# config의 hidden_size 확인 및 설정
print("Original hidden size:", config.hidden_size)
# 모델 초기화
model = LlamaBiPoolingModel.from_pretrained(
    model_name,
    config=config,
)
model.eval()  # 평가 모드로 설정
# 샘플 텍스트
texts = ["이것은 첫 번째 문장입니다.", "이것은 두 번째 문장이고 조금 더 깁니다."]

# 토크나이징
encoded = tokenizer(
    texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
)


# 모델 실행
with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
    )
