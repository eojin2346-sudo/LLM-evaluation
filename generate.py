# =============================================================
# generate.py — 모델별 응답 생성 (Groq 전용)
# =============================================================

import json
import os
from groq import Groq
import llm_config

# API 초기화
groq_client = Groq(api_key=llm_config.GROQ_API_KEY)

# ── 프롬프트 전략 ──────────────────────────────────────────────
PROMPT_STRATEGIES = {
    "zero_shot": lambda q: q,
    "few_shot":  lambda q: (
        "예시:\n"
        "Q: 지구에서 달까지 거리는?\n"
        "A: 평균 약 38만 4천 km입니다.\n\n"
        f"Q: {q}\nA:"
    ),
    "cot": lambda q: f"{q}\n\n단계적으로 생각해봅시다:",
}

# ── Groq 응답 함수 ─────────────────────────────────────────────
def call_groq(prompt: str, model_id: str) -> str:
    response = groq_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_config.TEMPERATURE,
        max_tokens=llm_config.MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

def get_response(model_name: str, prompt: str) -> str:
    groq_id = llm_config.GROQ_MODEL_IDS[model_name]
    return call_groq(prompt, groq_id)

# ── 전체 응답 생성 ─────────────────────────────────────────────
def generate_all(questions: list) -> list:
    results = []
    total = len(questions) * len(llm_config.MODELS) * len(llm_config.STRATEGIES)
    count = 0

    for q in questions:
        for model_name in llm_config.MODELS:
            for strategy in llm_config.STRATEGIES:
                count += 1
                prompt = PROMPT_STRATEGIES[strategy](q["question"])
                print(f"[{count}/{total}] {model_name} | {strategy} | Q{q['id']}")

                try:
                    response = get_response(model_name, prompt)
                except Exception as e:
                    print(f"  오류: {e}")
                    response = f"ERROR: {e}"

                results.append({
                    "question_id": q["id"],
                    "question":    q["question"],
                    "category":    q.get("category", "general"),
                    "model":       model_name,
                    "strategy":    strategy,
                    "prompt":      prompt,
                    "response":    response,
                })

    return results

# ── 단독 실행 ──────────────────────────────────────────────────
if __name__ == "__main__":
    with open(llm_config.QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"질문 {len(questions)}개 로드 완료\n")
    results = generate_all(questions)

    os.makedirs(llm_config.RESULTS_DIR, exist_ok=True)
    with open(llm_config.RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"응답 {len(results)}개 저장 완료 -> {llm_config.RESPONSES_PATH}")
