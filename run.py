# =============================================================
# run.py — 전체 파이프라인 실행
# 사용법: python run.py
# =============================================================

import json
import os
import llm_config
from generate import generate_all
from judge import run_single_judge, run_pairwise_judge
from evaluate import summarize_single, summarize_pairwise, load_scores, load_pairwise
from visualize import run as run_visualize
import pandas as pd

def main():
    os.makedirs(llm_config.RESULTS_DIR, exist_ok=True)

    # 1. 질문 로드
    print("="*60)
    print("1단계: 질문 로드")
    print("="*60)
    with open(llm_config.QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"질문 {len(questions)}개 / 모델 {len(llm_config.MODELS)}개 / 전략 {len(llm_config.STRATEGIES)}개")
    print(f"총 실험 수: {len(questions) * len(llm_config.MODELS) * len(llm_config.STRATEGIES)}개\n")

    # 2. 응답 생성
    print("="*60)
    print("2단계: 모델별 응답 생성")
    print("="*60)
    responses = generate_all(questions)
    with open(llm_config.RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"응답 {len(responses)}개 저장 완료\n")

    # 3. Single Judge 채점
    print("="*60)
    print("3단계: Single Judge 채점")
    print("="*60)
    responses = run_single_judge(responses)
    with open(llm_config.SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print("Single 채점 완료\n")

    # 4. Pairwise Judge 채점 (실행시 주석 제거)
    '''
    print("="*60)
    print("4단계: Pairwise Judge 채점")
    print("="*60)
    pairwise = run_pairwise_judge(responses)
    with open(llm_config.PAIRWISE_PATH, "w", encoding="utf-8") as f:
        json.dump(pairwise, f, ensure_ascii=False, indent=2)
    print("Pairwise 채점 완료\n")
    '''
    # 5. 결과 요약
    print("="*60)
    print("5단계: 결과 요약")
    print("="*60)
    df_single   = pd.DataFrame(responses)
    #df_pairwise = pd.DataFrame(pairwise)
    summarize_single(df_single)
    #summarize_pairwise(df_pairwise)

    # 6. 시각화
    print("\n" + "="*60)
    print("6단계: 시각화")
    print("="*60)
    run_visualize()

    print("\n실험 완료!")
    print(f"결과 폴더: {llm_config.RESULTS_DIR}/")

if __name__ == "__main__":
    main()
