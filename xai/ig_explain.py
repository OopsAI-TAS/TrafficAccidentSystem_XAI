# xai/ig_explain.py

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from captum.attr import LayerIntegratedGradients
import json

# 학습 때 썼던 TextOnlyHead 그대로 재사용
from train.train import TextOnlyHead, MAX_LEN, MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_DIR = Path("train/artifacts")

VALID_PATH = "train/valid.jsonl"

def load_sample_from_valid(idx: int = 0, path: str = VALID_PATH):
    """
    valid.jsonl에서 idx번째 샘플 하나 읽어서 (text, A) 리턴
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    row = rows[idx]
    return row["text"], row["A"]

def load_model_and_tokenizer():
    """
    train/train.py에서 저장한 tokenizer, bert, model.pt 로드
    """
    tok = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
    bert = AutoModel.from_pretrained(ARTIFACT_DIR)

    model = TextOnlyHead(bert)
    state = torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return tok, model

def forward_func(model, input_ids, attention_mask):
    """
    Captum이 호출할 forward 함수.
    input: input_ids, attention_mask
    output: 과실비율 predA (배치 단위)
    """
    return model(input_ids=input_ids, attention_mask=attention_mask)

def compute_ig_attributions(text, n_steps: int = 50):
    """
    하나의 텍스트에 대해 Integrated Gradients로
    토큰별 기여도(attribution)를 계산해서 (tokens, scores) 리턴
    (raw 토큰 / raw 점수, 후처리는 따로)
    """
    tok, model = load_model_and_tokenizer()

    # 1) 토크나이즈
    enc = tok(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)            # [1, L]
    attention_mask = enc["attention_mask"].to(DEVICE)  # [1, L]

    # 2) baseline 설정: 전부 [PAD] + attention_mask=0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    baseline_ids = torch.full_like(input_ids, pad_id).to(DEVICE)
    baseline_mask = torch.zeros_like(attention_mask).to(DEVICE)  # noqa: F841 (설명용)

    # 3) LayerIntegratedGradients 설정 (BERT embedding 레이어 기준)
    lig = LayerIntegratedGradients(
        lambda ids, mask: forward_func(model, ids, mask),
        model.bert.embeddings,
    )

    # 4) IG 계산
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=attention_mask,
        n_steps=n_steps,
    )  # shape: [1, L, H]

    # 5) hidden 차원 합쳐서 토큰별 스칼라 score로 축약
    token_attrs = attributions.sum(dim=-1).squeeze(0)  # [L]

    # 6) normalize (선택) – 절댓값 합이 1이 되도록
    token_attrs = token_attrs / (token_attrs.abs().sum() + 1e-8)

    # 7) 토큰 문자열로 변환
    tok_ids = input_ids[0]
    tokens = tok.convert_ids_to_tokens(tok_ids)

    # CPU로 옮겨서 numpy 리스트로
    scores = token_attrs.detach().cpu().numpy().tolist()

    # raw 토큰/점수 그대로 리턴 (후처리는 따로)
    return tokens, scores

# ---------------------------
# 후처리: 워드피스 병합 + 스페셜 토큰 제거 + 재정규화
# ---------------------------

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def merge_wordpieces(tokens, scores):
    """
    WordPiece 토큰("##")을 원래 단어 단위로 합치고,
    스페셜 토큰([CLS]/[SEP]/[PAD])은 제거.
    점수는 조각 점수의 합으로 계산.
    """
    merged_tokens = []
    merged_scores = []

    cur_token = ""
    cur_score = 0.0

    for t, s in zip(tokens, scores):
        # 스페셜 토큰은 아예 스킵
        if t in SPECIAL_TOKENS:
            continue

        if t.startswith("##"):
            sub = t[2:]
            # 앞에 이어 붙일 토큰이 없으면 그냥 새로 시작
            if cur_token == "":
                cur_token = sub
                cur_score = s
            else:
                cur_token += sub
                cur_score += s
        else:
            # 새 토큰 시작 전, 기존 거 flush
            if cur_token != "":
                merged_tokens.append(cur_token)
                merged_scores.append(cur_score)
            cur_token = t
            cur_score = s

    # 마지막 토큰 flush
    if cur_token != "":
        merged_tokens.append(cur_token)
        merged_scores.append(cur_score)

    return merged_tokens, merged_scores


def normalize_scores(scores):
    """
    절댓값 합이 1이 되도록 재정규화.
    (스페셜 토큰 제거/병합 이후에 다시 normalize)
    """
    denom = sum(abs(s) for s in scores) + 1e-8
    return [s / denom for s in scores]


def pretty_ig(text, n_steps: int = 50):
    """
    텍스트 하나를 넣으면:
    - raw WordPiece 토큰/점수
    - 병합된 token/점수(재정규화)
    둘 다 리턴해주는 헬퍼.
    """
    raw_tokens, raw_scores = compute_ig_attributions(text, n_steps=n_steps)
    merged_tokens, merged_scores = merge_wordpieces(raw_tokens, raw_scores)
    merged_scores = normalize_scores(merged_scores)
    return (raw_tokens, raw_scores), (merged_tokens, merged_scores)

TARGET_FEATURE_KEYS = ["유형코드", "장소", "장소특징", "A진행", "B진행"]


def aggregate_feature_attributions(tokens, scores):
    """
    병합된 토큰/점수(merged_tokens, merged_scores)를 받아서
    '유형코드', '장소', '장소특징', 'A진행', 'B진행' 단위로 IG를 합산.

    feature span은:
      - 시작: 해당 key 토큰 등장 위치
      - 끝: 다음 중 먼저 나오는 것 직전까지
        * ','  (필드 구분)
        * '['  (새 섹션 시작, 예: [차량])
        * 다른 TARGET_FEATURE_KEYS
    """
    features = {}
    n = len(tokens)
    i = 0

    while i < n:
        t = tokens[i]
        if t in TARGET_FEATURE_KEYS:
            # 현재 feature key (예: "장소특징")
            key_name = t

            j = i
            span_tokens = []
            span_scores = []

            while j < n:
                tj = tokens[j]

                # 현재 key 이후에 다시 다른 key가 나오면 거기서 끊기
                if j > i and tj in TARGET_FEATURE_KEYS:
                    break
                # 섹션 구분자 or 필드 구분자 만나면 끊기
                if tj in [",", "["]:
                    break

                span_tokens.append(tj)
                span_scores.append(scores[j])
                j += 1

            # "유형코드=214" 같은 식으로 feature 텍스트 구성
            feature_text = "".join(span_tokens)
            feat_key = feature_text

            feat_score = sum(span_scores)
            features[feat_key] = feat_score

            i = j  # span 끝으로 점프
        else:
            i += 1

    # 정규화 (선택): 절댓값 합 1
    denom = sum(abs(v) for v in features.values()) + 1e-8
    features = {k: v / denom for k, v in features.items()}

    return features

if __name__ == "__main__":
    # 1) val에서 한 샘플 가져오기
    text, true_A = load_sample_from_valid(idx=0)
    print(f"[sample] true A = {true_A}")
    print(f"[sample] text =\n{text}\n")

    # 2) 모델 예측값도 같이 찍기 (진짜 학습된 모델 쓰는지 확인용)
    tok, model = load_model_and_tokenizer()
    enc = tok(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        predA = model(input_ids=input_ids, attention_mask=attention_mask).item()
    print(f"[sample] pred A = {predA:.2f}\n")

    # 3) IG (토큰 레벨 → word 레벨)
    (raw_tokens, raw_scores), (merged_tokens, merged_scores) = pretty_ig(
        text, n_steps=50
    )

    # 4) feature 단위로 집계
    feat_scores = aggregate_feature_attributions(merged_tokens, merged_scores)

    print("=== Feature-level IG attributions ===")
    for k, v in feat_scores.items():
        print(f"{k:35s}  {v:+.4f}")