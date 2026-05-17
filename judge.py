# =============================================================
# judge.py — Single + Pairwise LLM-as-a-Judge (Groq 전용)
# =============================================================

import json
import re
from groq import Groq
import llm_config

# API 초기화
groq_client = Groq(api_key=llm_config.GROQ_API_KEY)

# ── Judge 프롬프트 로드 ────────────────────────────────────────
def load_judge_prompts() -> dict:
    prompts = {}
    with open(llm_config.JUDGE_PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                prompts[p["name"]] = p
    return prompts

JUDGE_PROMPTS = load_judge_prompts()

# ── Judge 모델 호출 ────────────────────────────────────────────
def call_judge(system_prompt: str, user_prompt: str) -> str:
    groq_id = llm_config.GROQ_MODEL_IDS[llm_config.JUDGE_MODEL]
    response = groq_client.chat.completions.create(
        model=groq_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# ── Single Judge: 응답 1개 채점 ────────────────────────────────
def judge_single(question: str, answer: str, prompt_name: str = "single-ko-v1") -> dict:
    p = JUDGE_PROMPTS[prompt_name]
    user_prompt = p["prompt_template"].format(question=question, answer=answer)

    try:
        raw = call_judge(p["system_prompt"], user_prompt)
        match = re.search(r"\[\[(\d+)\]\]", raw)
        score = int(match.group(1)) if match else 0
        return {"score": score, "reason": raw, "judge_type": "single", "prompt_name": prompt_name}
    except Exception as e:
        return {"score": 0, "reason": f"ERROR: {e}", "judge_type": "single", "prompt_name": prompt_name}

# ── Pairwise Judge: 두 응답 비교 ──────────────────────────────
def judge_pairwise(question: str, answer_a: str, answer_b: str,
                   model_a: str, model_b: str,
                   prompt_name: str = "pair-ko-v1") -> dict:
    p = JUDGE_PROMPTS[prompt_name]
    user_prompt = p["prompt_template"].format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )

    try:
        raw = call_judge(p["system_prompt"], user_prompt)
        match = re.search(r"\[\[([ABC])\]\]", raw)
        verdict = match.group(1) if match else "?"
        winner = model_a if verdict == "A" else (model_b if verdict == "B" else "tie")
        return {
            "model_a":     model_a,
            "model_b":     model_b,
            "verdict":     verdict,
            "winner":      winner,
            "reason":      raw,
            "judge_type":  "pairwise",
            "prompt_name": prompt_name,
        }
    except Exception as e:
        return {
            "model_a": model_a, "model_b": model_b,
            "verdict": "?", "winner": "error",
            "reason": f"ERROR: {e}", "judge_type": "pairwise", "prompt_name": prompt_name,
        }

# ── 전체 Single 채점 ──────────────────────────────────────────
def run_single_judge(responses: list) -> list:
    total = len(responses)
    for i, row in enumerate(responses):
        print(f"  [Single {i+1}/{total}] {row['model']} | {row['strategy']} | Q{row['question_id']}")
        result = judge_single(row["question"], row["response"])
        row.update(result)
    return responses

# ── 전체 Pairwise 채점 ────────────────────────────────────────
def run_pairwise_judge(responses: list) -> list:
    from itertools import combinations

    groups = {}
    for row in responses:
        key = (row["question_id"], row["strategy"])
        groups.setdefault(key, []).append(row)

    results = []
    total_pairs = sum(len(list(combinations(g, 2))) for g in groups.values())
    count = 0

    for (qid, strategy), rows in groups.items():
        for row_a, row_b in combinations(rows, 2):
            count += 1
            print(f"  [Pairwise {count}/{total_pairs}] Q{qid} | {strategy} | {row_a['model']} vs {row_b['model']}")
            result = judge_pairwise(
                question=row_a["question"],
                answer_a=row_a["response"],
                answer_b=row_b["response"],
                model_a=row_a["model"],
                model_b=row_b["model"],
            )
            result.update({
                "question_id": qid,
                "question":    row_a["question"],
                "strategy":    strategy,
            })
            results.append(result)

    return results

# ── 단독 실행 ──────────────────────────────────────────────────
if __name__ == "__main__":
    with open(llm_config.RESPONSES_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)

    print("=== Single Judge 채점 시작 ===")
    responses = run_single_judge(responses)
    with open(llm_config.SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"Single 채점 완료 -> {llm_config.SCORES_PATH}")

    # Pairwise 사용 시 아래 주석 해제
    # print("=== Pairwise Judge 채점 시작 ===")
    # pairwise = run_pairwise_judge(responses)
    # with open(llm_config.PAIRWISE_PATH, "w", encoding="utf-8") as f:
    #     json.dump(pairwise, f, ensure_ascii=False, indent=2)
    # print(f"Pairwise 채점 완료 -> {llm_config.PAIRWISE_PATH}")
