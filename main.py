#!/usr/bin/env python3
import os, sys, json, argparse, time
from dotenv import load_dotenv
from pydantic import ValidationError
from schema import AccidentPayload
from eval_schema import EvalResult
from prompts import build_messages, build_gemini_eval_prompt

# ── 환경 로드 ────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ .env에 OPENAI_API_KEY가 필요합니다.", file=sys.stderr)
    sys.exit(1)

# ── SDK 클라이언트 ──────────────────────────────────────────────────────────
from openai import OpenAI

gpt_client = OpenAI(api_key=OPENAI_API_KEY)

# ── 유틸 ────────────────────────────────────────────────────────────────────
def read_json_file(path: str) -> AccidentPayload:
    if not os.path.exists(path):
        print(f"❌ 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AccidentPayload.model_validate(data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}", file=sys.stderr)
        sys.exit(3)
    except ValidationError as e:
        print("❌ 스키마 검증 실패:\n" + str(e), file=sys.stderr)
        sys.exit(4)

def call_gpt_generate(payload: AccidentPayload):
    messages = build_messages(payload.model_dump())
    t0 = time.time()
    resp = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    t1 = time.time()
    content = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    return content, (t1 - t0), (usage.model_dump() if usage else None)

def call_gpt_evaluate(gpt_document: str, input_json: dict):
    prompt = build_gemini_eval_prompt(gpt_document, input_json)
    t0 = time.time()
    resp = gpt_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "You are a strict rubric grader. Output JSON only."},
         {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    t1 = time.time()
    raw = (resp.choices[0].message.content or "").strip()

    # JSON 파싱 & 검증
    parsed = None
    err = None
    try:
        parsed = json.loads(raw)
        EvalResult.model_validate(parsed)  # 스키마 검증
    except Exception as e:
        err = f"평가 JSON 파싱/검증 실패: {e}"

    return raw, parsed, (t1 - t0), err

# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Accident judgment: file → GPT(doc) → Gemini(eval)")
    parser.add_argument("--file", "-f", required=True, help="입력 JSON 파일 경로")
    args = parser.parse_args()

    payload = read_json_file(args.file)
    input_json = payload.model_dump()
    print(f"✅ 입력 파일 로드: {args.file}")

    # 1) 판결문 생성
    print("🧠 판결문 생성중...")
    gpt_doc, gpt_secs, gpt_usage = call_gpt_generate(payload)
    print(f"✅ 판결문 완료 ({gpt_secs:.2f}s)\n")

    # 2) 루브릭 평가(JSON)
    print("🧮 루브릭 평가중...")
    gemini_raw, gemini_parsed, gemini_secs, gemini_err = call_gpt_evaluate(gpt_doc, input_json)
    if gemini_err:
        print(f"⚠️ {gemini_err}", file=sys.stderr)
    print(f"✅ 평가 완료 ({gemini_secs:.2f}s)\n")

    # 콘솔 출력 (요약)
    print("\n================ [ 판결문 ] ================\n")
    print(gpt_doc)
    print("\n================ [ 평가 결과 ] ================\n")
    print(gemini_raw)

    # 결과 저장
    out_base = os.path.splitext(args.file)[0]
    result_path = out_base + "_result.json"
    result = {
        "models": {"gpt": GPT_MODEL, "gemini": GEMINI_MODEL},
        "timing_sec": {"gpt": gpt_secs, "gemini": gemini_secs},
        "usage": {"gpt": gpt_usage},
        "outputs": {
            "gpt_document": gpt_doc,
            "gemini_eval_raw": gemini_raw,
            "gemini_eval_parsed": gemini_parsed,  # 파싱 성공 시 dict, 아니면 null
        },
        "errors": {"gemini_parse": gemini_err},
        "input_file": os.path.abspath(args.file),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장: {result_path}")

if __name__ == "__main__":
    main()
