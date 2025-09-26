# TrafficAccidentSystem_LLM 🚦

교통사고 블랙박스 영상을 기반으로 **RAG (Retrieval-Augmented Generation)** 기법을 활용해  
자동으로 **과실비율, 근거, 적용 법조문, 참고 판례**를 생성하는 시스템입니다.

---

## 📌 프로젝트 개요
- **입력**: CV 모듈에서 추출된 사고 메타데이터 (`input.json`)  
- **검색**: FAISS 기반 벡터 DB에서 관련 **기본과실표, 조정예시, 법조문, 판례** 검색  
- **생성**: LLM이 검색 결과를 바탕으로 최종 **과실비율(총합 100%)과 근거** 작성  
- **출력**: 사람이 읽기 쉬운 `answer.txt` 와 API 연동용 `answer.json` 제공  

---

## ⚙️ 실행 방법

### 1. 가상환경 & 패키지 설치
```bash
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. 전체 실행 순서
```bash
# 1) 인덱스 구축 (법조문/판례/과실기준)
PYTHONPATH=. python3 rag/build_index.py

# 2) 입력 요약 + 질의 생성
PYTHONPATH=. python3 rag/query_builder.py

# 3) 검색 + 프롬프트 생성
PYTHONPATH=. python3 rag/query.py

# 4) LLM 호출 → answer.txt 생성
PYTHONPATH=. python3 rag/answer.py

# 5) 결과 JSON 변환 → answer.json 생성
PYTHONPATH=. python3 rag/convert_answer_to_json.py
```

### 4 결과물 확인
	•	data/index/answer.txt : 사람이 읽기 좋은 리포트
	•	data/index/answer.json : API/후처리용 JSON

### 5 핵심 요약
1.	build_index.py → 법조문/판례/과실기준 데이터를 청크화 후 FAISS 벡터 DB 생성
	2.	retriever.py → 입력 메타데이터와 가장 유사한 조항·판례 검색
	3.	answer.py → 검색된 컨텍스트를 프롬프트에 포함해 LLM 호출
	•	기본과실표/조정예시/법조문/판례 모두 강제 포함
	•	A/B 합이 100%가 아니면 자동 정규화
	4.	convert_answer_to_json.py → 최종 텍스트 결과를 JSON 포맷으로 변환

### 6 폴더 구조
```bash
data/
  law_json/         # 과실기준/법조문/판례 원천 데이터
  mappings/         # 코드 → 라벨 매핑
  index/            # 실행 결과물(answer.txt, answer.json 등)
rag/
  samples/          # 입력 샘플 (input.json)
  *.py              # 파이프라인 코드
```



