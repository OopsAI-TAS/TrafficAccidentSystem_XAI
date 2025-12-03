# xai/ig_explain.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json

#############################################
# 0) 모델 + 토크나이저 로드 (Regression 버전)
#############################################

CKPT = Path("train/artifacts")

# ★ 중요: 학습때 쓴 구조와 똑같아야 하며, IG를 위해 inputs_embeds를 받아야 함
class TrafficRegressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        # 1차원 출력 (Regression)
        self.regressor = nn.Linear(bert.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # 1) IG 계산을 위해 inputs_embeds 지원 추가
        if inputs_embeds is not None:
            out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 2) CLS 토큰
        h_cls = out.last_hidden_state[:, 0]
        h_cls = self.dropout(h_cls)
        
        # 3) 회귀 예측
        predA = self.regressor(h_cls)
        
        # [Batch, 1] -> [Batch]
        return predA.squeeze(-1)

def load_model_and_tokenizer():
    # 저장된 아티팩트 경로
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    bert = AutoModel.from_pretrained(str(CKPT))

    # 모델 초기화
    model = TrafficRegressor(bert)
    
    # 가중치 로드
    state = torch.load(CKPT / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tok, model


#############################################
# 1) Embedding-Space IG (기존 로직 유지)
#############################################

def integrated_gradients_embedding(model, tokenizer, text, n_steps=50):
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True
    )

    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    # 1) Word embedding
    embed_layer = model.bert.embeddings.word_embeddings
    input_embed = embed_layer(input_ids)

    # baseline: zero embed
    baseline = torch.zeros_like(input_embed)
    total_grad = torch.zeros_like(input_embed)
    alphas = torch.linspace(0, 1, n_steps).to(device)

    for alpha in alphas:
        x = baseline + alpha * (input_embed - baseline)
        x = x.detach().clone().requires_grad_(True)
        x.retain_grad()

        # 모델 forward (inputs_embeds 사용)
        predA = model(inputs_embeds=x, attention_mask=mask)
        
        model.zero_grad()
        predA.backward(retain_graph=True)

        total_grad += x.grad

    ig_embed = (input_embed - baseline) * (total_grad / n_steps)
    token_ig = ig_embed.sum(dim=-1).detach().cpu().numpy().tolist()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, token_ig


#############################################
# 2) Helper Functions (기존 유지)
#############################################

def merge_wordpiece(tokens, scores):
    merged_tokens = []
    merged_scores = []
    current_token = ""
    current_score = 0

    for tok, sc in zip(tokens, scores):
        if tok.startswith("##"):
            current_token += tok[2:]
            current_score += sc
        else:
            if current_token != "":
                merged