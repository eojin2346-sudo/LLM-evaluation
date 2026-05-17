# =============================================================
# load_dataset.py — 한국어 벤치마크 데이터 로드
# KMMLU 45개 과목에서 supercategory 균형 샘플링 → questions.json
#
# 총 75개 질문 (15개 subset × 5문제)
# API 호출: 응답 675건(Groq) + Judge 675건(Gemini) = 1,350건 ✅
#
# Supercategory 배분:
#   HUMSS           → 4개 subset × 5문제 = 20개
#   STEM            → 4개 subset × 5문제 = 20개
#   Applied Science → 4개 subset × 5문제 = 20개
#   Other           → 3개 subset × 5문제 = 15개
#
# Human Accuracy: CSV의 "Human Accuracy" 컬럼 그대로 저장
#   → 모델 정답률 vs 인간 정답률 직접 비교 가능
#   → 약 90% 문항에 존재, 없으면 None
#
# Truncation 정책:
#   질문을 자르지 않음. 선택지가 잘리면 benchmark가 깨지기 때문.
#   600자 초과 시 경고만 출력 → 해당 subset 교체 여부를 수동 판단.
#
# 사용법: python load_dataset.py
# =============================================================

import json
import os
import random
from datasets import load_dataset

random.seed(42)

# ── 설정 ──────────────────────────────────────────────────────
SAMPLE_CONFIG = [
    # ── HUMSS (인문사회) ───────────────────────────────────────
    {"subset": "Accounting",     "category": "accounting",     "super": "HUMSS", "n": 5},
    {"subset": "Law",            "category": "law",            "super": "HUMSS", "n": 5},
    {"subset": "Economics",      "category": "economics",      "super": "HUMSS", "n": 5},
    {"subset": "Korean-History", "category": "korean_history", "super": "HUMSS", "n": 5},

    # ── STEM ──────────────────────────────────────────────────
    {"subset": "Computer-Science", "category": "computer_science", "super": "STEM", "n": 5},
    {"subset": "Biology",          "category": "biology",          "super": "STEM", "n": 5},
    {"subset": "Chemistry",        "category": "chemistry",        "super": "STEM", "n": 5},
    {"subset": "Math",             "category": "math",             "super": "STEM", "n": 5},

    # ── Applied Science (응용과학) ─────────────────────────────
    {"subset": "Information-Technology", "category": "information_technology", "super": "Applied Science", "n": 5},
    {"subset": "Electrical-Engineering", "category": "electrical_engineering", "super": "Applied Science", "n": 5},
    {"subset": "Environmental-Science",  "category": "environmental_science",  "super": "Applied Science", "n": 5},
    {"subset": "Energy-Management",      "category": "energy_management",      "super": "Applied Science", "n": 5},

    # ── Other ─────────────────────────────────────────────────
    {"subset": "Health",        "category": "health",        "super": "Other", "n": 5},
    {"subset": "Real-Estate",   "category": "real_estate",   "super": "Other", "n": 5},
    {"subset": "Public-Safety", "category": "public_safety", "super": "Other", "n": 5},
]

DATASET_NAME   = "HAERAE-HUB/KMMLU"
SPLIT          = "test"
OUTPUT_PATH    = "data/questions.json"
WARN_LEN       = 600   # 이 길이 초과 시 경고 출력 (자르지는 않음)

# ── KMMLU 포맷 파싱 ───────────────────────────────────────────
def parse_kmmlu(row: dict, cfg: dict, qid: int) -> dict:
    """
    KMMLU CSV 컬럼: question, answer, A, B, C, D, Category, Human Accuracy

    Truncation 없음:
      선택지(A~D)가 잘리면 객관식 benchmark 자체가 깨지므로 자르지 않는다.
      대신 WARN_LEN 초과 시 경고를 출력하고, 해당 subset 교체를 수동 검토한다.

    Human Accuracy:
      실제 시험 응시자 정답률 (0.0 ~ 1.0). 없으면 None.
    """
    question = row["question"]
    choices  = [row.get(k, "") for k in ["A", "B", "C", "D"]]
    choices_str = "\n".join(
        f"  {chr(65 + i)}) {c}" for i, c in enumerate(choices) if c
    )

    full_question = f"{question}\n\n보기:\n{choices_str}"

    # 길이 경고 (truncate 하지 않음)
    if len(full_question) > WARN_LEN:
        print(
            f"  ⚠ 긴 질문 감지 ({len(full_question)}자): "
            f"Q{qid} [{cfg['subset']}] — subset 교체 검토 필요"
        )

    # Human Accuracy
    raw_ha = row.get("Human Accuracy", None)
    try:
        human_accuracy = float(raw_ha) if raw_ha is not None else None
    except (ValueError, TypeError):
        human_accuracy = None

    return {
        "id":            qid,
        "question":      full_question,
        "category":      cfg["category"],
        "supercategory": cfg["super"],
        "subset":        cfg["subset"],
        "source":        DATASET_NAME,
        "answer":        row.get("answer", ""),
        "human_accuracy": human_accuracy,
        "original_data": {
            "question": question,
            "choices":  choices,
        },
    }

# ── 데이터 로드 ───────────────────────────────────────────────
def load_samples() -> list:
    all_samples = []
    qid = 1

    for cfg in SAMPLE_CONFIG:
        print(f"로드 중: [{cfg['super']:15s}] {cfg['subset']} ({cfg['n']}개)")

        try:
            ds = load_dataset(DATASET_NAME, cfg["subset"], split=SPLIT)
        except Exception as e:
            print(f"  ⚠ 로드 실패: {e}")
            continue

        indices = random.sample(range(len(ds)), min(cfg["n"], len(ds)))

        for idx in indices:
            sample = parse_kmmlu(ds[idx], cfg, qid)
            all_samples.append(sample)
            qid += 1

        print(f"  ✓ {len(indices)}개 완료")

    return all_samples

# ── 저장 ──────────────────────────────────────────────────────
def save(samples: list):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n총 {len(samples)}개 질문 저장 완료 → {OUTPUT_PATH}")

# ── 미리보기 ──────────────────────────────────────────────────
def preview(samples: list, n: int = 3):
    print("\n--- 샘플 미리보기 ---")
    for s in samples[:n]:
        ha = f"{s['human_accuracy']:.1%}" if s["human_accuracy"] is not None else "N/A"
        print(f"[Q{s['id']}] [{s['supercategory']}] [{s['subset']}]")
        print(f"      {s['question'][:80]}...")
        print(f"      정답: {s['answer']}  |  인간 정답률: {ha}\n")

# ── 통계 출력 ─────────────────────────────────────────────────
def print_stats(samples: list):
    from collections import Counter

    print("\n[Supercategory별]")
    for cat, cnt in Counter(s["supercategory"] for s in samples).items():
        print(f"  {cat:20s}: {cnt}개")

    # 길이 분포
    lengths = [len(s["question"]) for s in samples]
    over    = [s for s in samples if len(s["question"]) > WARN_LEN]
    print(f"\n[질문 길이]")
    print(f"  평균: {sum(lengths) // len(lengths)}자")
    print(f"  최대: {max(lengths)}자")
    print(f"  {WARN_LEN}자 초과: {len(over)}개"
          + (f" → {[s['subset'] for s in over]}" if over else ""))

    # Human Accuracy
    ha_list = [s["human_accuracy"] for s in samples if s["human_accuracy"] is not None]
    print(f"\n[Human Accuracy]")
    print(f"  커버리지: {len(ha_list)}/{len(samples)}개 ({len(ha_list)/len(samples):.1%})")
    if ha_list:
        print(f"  평균: {sum(ha_list)/len(ha_list):.1%}")
        print(f"  범위: {min(ha_list):.1%} ~ {max(ha_list):.1%}")

    total        = len(samples)
    groq_calls   = total * 3
    gemini_calls = groq_calls
    print(f"\n합계: {total}개")
    print(f"예상 API 호출: 응답 {groq_calls}건(Groq) + Judge {gemini_calls}건(Gemini) = {groq_calls + gemini_calls}건")

# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("KMMLU 데이터 로드 (supercategory 균형 샘플링)")
    print(f"목표: 75개 / 15개 subset × 5문제 / Gemini 한도 1,500건 이내")
    print("=" * 60)

    samples = load_samples()
    preview(samples)
    save(samples)
    print_stats(samples)
