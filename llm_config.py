# =============================================================
# config.py — 실험 설정값 모음 (Groq 전용)
# =============================================================

import os

# ── API 키 ────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── 응답 생성 모델 ─────────────────────────────────────────────
MODELS = {
    "llama-4-scout": "groq",    # gemini-2.5-flash 대체
    "llama-3.1-8b":  "groq",
    "llama-3.3-70b": "groq",    # mixtral-8x7b 대체
}

# Groq 모델 ID 매핑
GROQ_MODEL_IDS = {
    "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b":  "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

# ── Judge 모델 ────────────────────────────────────────────────
JUDGE_MODEL    = "llama-3.3-70b"
JUDGE_PROVIDER = "groq"

# ── 프롬프트 전략 ──────────────────────────────────────────────
STRATEGIES = ["zero_shot", "few_shot", "cot"]

# ── 파일 경로 ─────────────────────────────────────────────────
QUESTIONS_PATH     = "data/questions.json"
JUDGE_PROMPTS_PATH = "data/judge_prompts.jsonl"
RESPONSES_PATH     = "data/results/responses.json"
SCORES_PATH        = "data/results/scores.json"
PAIRWISE_PATH      = "data/results/pairwise.json"
RESULTS_DIR        = "data/results"

# ── 실험 옵션 ─────────────────────────────────────────────────
MAX_TOKENS  = 512
TEMPERATURE = 0.0
