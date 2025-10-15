# TrafficAccidentSystem_XAI
---

## ⚖️ 프로젝트 소개

**OopsAI**는 교통사고 영상을 분석하고, 그에 대한 **법률 판결문을 자동 생성하고 평가하는** AI 시스템입니다.

2025학년도 2학기 **건국대학교 드림학기제(자기설계학기제)** 프로젝트로 진행되며,
생성형 AI의 **설명가능성(Explainability)**과 **신뢰성(Trustworthiness)**을 실험적으로 검증하는 것을 목표로 합니다.

---

## 👥 팀 구성

| 이름      | 역할                           |
| ------- | ---------------------------- |
| 윤서진     | 교통사고 영상 분석            |
| 조은영     | LLM      |
| 송은서 | XAI |

---

## 📦 Requirements

* Python 3.10+
* 패키지:

  * `openai`
  * `python-dotenv`
  * `pydantic`
* 로컬 모듈:

  * `schema.py` (AccidentPayload)
  * `eval_schema.py` (EvalResult)
  * `prompts.py` (`build_messages`, `build_gemini_eval_prompt`)

### 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install openai pydantic python-dotenv
```

## 🔐 Environment

루트 디렉터리에 `.env` 파일을 만들고 OpenAI API 키를 넣어주세요.

```
OPENAI_API_KEY=sk-...
```

## ▶️ Usage

```bash
python main.py --file path/to/input.json
# 또는
python main.py -f path/to/input.json
```