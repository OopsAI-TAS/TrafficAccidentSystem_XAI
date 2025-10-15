import json
import random
from datetime import datetime

# ── GPT용 메시지 빌더 (사용자 제공 템플릿 반영) ─────────────────────────────
def build_messages(input_json):
    summary = input_json["summary"]
    reasons = input_json["reasons"]
    laws = input_json["laws"]
    precedents = input_json["precedents"]
    meta = input_json["input"]

    year = datetime.now().year
    serial = random.randint(10000, 99999)

    system = "\n".join([
        "당신은 교통사고 과실비율 사건의 판결문을 작성하는 한국 법조문체 전문가입니다.",
        "요구사항:",
        "1) 아래 템플릿을 반드시 따르십시오.",
        "2) 표제어는 굵게 하지 말고, 자연스러운 문단 구성을 유지하십시오.",
        "3) 숫자, 백분율, 날짜는 명확히 표기하고, 법조문과 판례는 간결히 인용하십시오.",
        "4) AI, 생성형, 모델 언급을 금지합니다.",
        "템플릿:",
        f"사건번호: {year}가단{serial}",
        f"사건일자: {meta['video_date']}",
        "",
        "[주문]",
        "1. 피사고 관계자 A의 과실은 ○○%이고, 관계자 B의 과실은 ○○%로 정한다.",
        "2. 이에 따른 손해배상 책임은 각자의 과실 비율에 따른다.",
        "",
        "[이유]",
        "1. 사실관계",
        "   - 사고 발생 일시 및 장소",
        "   - 차량 A의 진행 방향 및 신호 위반 여부",
        "   - 차량 B의 운행 상황 및 주의의무 이행 여부",
        "",
        "2. 법적 평가",
        "   - 도로교통법 제○○조 위반 여부",
        "   - 관련 판례와의 비교 검토",
        "   - 양측 행위의 주의의무 판단",
        "",
        "3. 결론",
        "   - A에게 주된 과실 인정",
        "   - B의 보조적 과실 반영",
        "   - 최종 과실 비율 산출",
    ])

    user = {
        "summary": input_json["summary"],
        "reasons": input_json["reasons"],
        "laws": input_json["laws"],
        "precedents": input_json["precedents"],
        "input": input_json["input"],
        "guidance": {
            "structure": ["사건번호", "사건일자", "[주문]", "[이유]"],
            "style": "간결하고 단정적인 판결문 문체. 800~1200자 내외.",
            "must_include": [
                "사건번호와 사건일자 명시",
                "주문 항목에 과실 비율과 손해배상 책임 명시",
                "이유 항목에 사실관계, 법적 평가, 결론 포함",
                "적용 법조문 및 판례의 핵심 취지 반영",
                "최종 과실비율 명시(A: xx%, B: yy%)",
            ],
        },
    }

    exemplar = """
사건번호: 2024가단12345
사건일자: 2024년 7월 15일  

[주문]  
1. 피사고 관계자 A의 과실은 60%이고, 관계자 B의 과실은 40%로 정한다.  
2. 이에 따른 손해배상 책임은 각자의 과실 비율에 따른다.  

[이유]  

1. 사실관계  
본 사건은 2024년 7월 15일 서울시 강남구 교차로에서 발생한 교통사고로, A 차량은 직진 중이었고 B 차량은 좌회전 중이었다. 사고 당시 A 차량은 신호를 준수하며 교차로를 통과하고 있었으며, B 차량은 좌회전을 시도하던 중 충돌이 발생하였다.  
현장 영상에 따르면, 사고 당시 교차로의 신호등은 정상적으로 작동하고 있었으며, A 차량은 제한 속도를 준수하고 있었다. 반면, B 차량은 좌회전 시 충분한 주의를 기울이지 않아 A 차량의 진행을 방해한 것으로 보인다.  

2. 법적 평가  
도로교통법 제13조에 따르면, 교차로에서의 통행방법을 준수할 의무가 있으며, 특히 좌회전 차량은 직진 차량에 대해 높은 주의의무를 부담한다. 관련 판례(대법원 2015다12345)에 따르면, 좌회전 차량의 주의의무 위반은 과실 비율 산정에 있어 중요한 요소로 작용한다.  
본 사건에서 B 차량은 좌회전 시 충분한 주의를 기울이지 않았으며, 이는 주의의무 위반으로 판단된다. 또한, A 차량은 직진 중이었으나 일부 속도 조절 의무를 다하지 못한 점이 인정된다.  
특히, B 차량의 좌회전 각도와 시야 확보의 어려움이 사고 발생에 기여한 점을 고려할 때, B 차량의 과실 비율은 기본 과실표에 따라 60%로 산정된다. 그러나, A 차량의 일부 과실이 인정되어 B 차량의 과실 비율을 10% 감소시키고, A 차량의 과실 비율을 10% 증가시키는 조정이 이루어진다.  

3. 결론  
위 사실관계와 법적 평가를 종합적으로 고려할 때, A 차량의 과실은 60%로, B 차량의 과실은 40%로 산정된다.  
결론적으로, 본 법원은 A 차량의 과실을 60%, B 차량의 과실을 40%로 최종 판단한다.  

본 판결은 도로교통법 제13조 및 관련 판례를 근거로 하였으며, 사고 당시의 현장 영상과 제출된 증거 자료를 종합적으로 검토한 결과에 따른 것이다.  
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
        {"role": "assistant", "content": exemplar.strip()},
    ]

# ── Gemini 평가 루브릭 프롬프트 ─────────────────────────────────────────────
def build_gemini_eval_prompt(gpt_document: str, input_json: dict) -> str:
    """
    gpt_document: GPT가 생성한 판결문(텍스트)
    input_json: 원 입력 JSON (summary/reasons/laws/precedents/input)
    Gemini는 반드시 JSON 하나로만 응답하도록 지시함.
    """
    rubric = """
[평가 루브릭] (가중치 합 100)
1) 구조·템플릿 준수 (15): 사건번호/사건일자/[주문]/[이유] 구조 충족, 굵은 표제어 사용 금지, 문단 구성 자연스러움.
2) 수치·형식 정확성 (20): A/B 최종 과실 % 명시, 합계 100% 여부, 날짜·백분율 표기 정확성.
3) 사실관계 충실도 (15): 제공된 JSON 내 사실(요약, base_table, adjustments, input)을 벗어나지 않음.
4) 법적 평가 적합성 (20): 제공된 법조문/판례만 간결 인용, 논리적 연결성.
5) 완결성 (15): 사실관계→법적평가→결론 흐름과 주문·결론의 정합성.
6) 명료성·문체 (10): 간결하고 단정적인 판결문 문체, 과장/추측 없음.
7) 비창작성·금지사항 준수 (5): 외부 사실 창작 금지, AI/모델 언급 금지.
"""

    schema_spec = """
[출력 형식(JSON, ONLY JSON)]
{
  "total_score": 0-100,                             // 가중 합산 정수
  "criteria": [
    {"name":"구조·템플릿 준수","weight":15,"score":0-100,"comments":"...", "evidence":["..."]},
    {"name":"수치·형식 정확성","weight":20,"score":0-100,"comments":"...","evidence":["..."]},
    {"name":"사실관계 충실도","weight":15,"score":0-100,"comments":"...","evidence":["..."]},
    {"name":"법적 평가 적합성","weight":20,"score":0-100,"comments":"...","evidence":["..."]},
    {"name":"완결성","weight":15,"score":0-100,"comments":"...","evidence":["..."]},
    {"name":"명료성·문체","weight":10,"score":0-100,"comments":"...","evidence":["..."]},
    {"name":"비창작성·금지사항 준수","weight":5,"score":0-100,"comments":"...","evidence":["..."]}
  ],
  "flags": {
    "hallucination": false,
    "missing_fields": [],           // 누락된 필드/문단
    "template_violations": [],      // 템플릿 위반(굵은 표제어 등)
    "math_errors": []               // 백분율 합계 오류 등
  },
  "suggestions": ["...","..."],     // 개선 제안
  "extracted": {
    "final_ratio": {"A": null, "B": null},
    "cited_laws": [],
    "cited_precedents": []
  }
}
※ 주의: 반드시 위 JSON 스키마로만 출력. 추가 텍스트 금지.
"""

    guidance = """
[평가 지침]
- 판단 근거는 오직 [입력 JSON]과 [GPT 판결문]에 한정.
- 외부 사실/조문/판례를 추가하지 말 것.
- 최종 과실비율이 명시되지 않으면 수치·형식 정확성에서 감점하고 flags.missing_fields에 기록.
- 백분율 합이 100이 아니면 flags.math_errors에 기록.
- 사건번호/사건일자/주문/이유 누락 시 flags.missing_fields에 기록.
- 문서에 굵은 표제어 사용 시 flags.template_violations에 기록.
"""

    prompt = f"""
당신은 교통사고 과실비율 판결문 평가자입니다. 아래 루브릭으로 [GPT 판결문]을 평가하십시오.
{rubric}

{guidance}

{schema_spec}

[입력 JSON]
{json.dumps(input_json, ensure_ascii=False, indent=2)}

[GPT 판결문]
{gpt_document}
"""
    return prompt.strip()