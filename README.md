# LLM-as-a-Judge 기반 한국어 응답 품질 자동 평가 시스템

> NeurIPS 2023 논문 "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"의 평가 파이프라인을 참고하여,  
> 한국어 벤치마크(KMMLU)에 맞게 재설계한 경량 실험 프레임워크입니다.

---

## 프로젝트 배경

FastChat(lm-sys)의 LLM-as-a-Judge 구조는 MT-Bench 영어 질문과 GPT-4 Judge를 기본값으로 사용합니다.  
본 프로젝트는 이 구조를 그대로 따르되, 아래 세 가지를 한국어 환경에 맞게 재설계했습니다.

| 항목 | FastChat 원본 | 본 프로젝트 |
|---|---|---|
| 평가 데이터 | MT-Bench (영어) | KMMLU (한국어, 75문항) |
| Judge 모델 | GPT-4 (유료) | LLaMA 3.3 70B via Groq (무료) |
| 응답 모델 | Vicuna 등 | LLaMA 3종 via Groq (무료) |
| Judge 방식 | Single grading | Single + Pairwise 둘 다 |
| 시각화 | 없음 | 히트맵 / 승률 막대 / 박스플롯 |

---

## 평가 데이터셋

[KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) — 한국 실제 시험에서 수집한 전문가 수준 한국어 벤치마크

| 카테고리 | 문항 수 |
|---|---|
| HUMSS (Accounting, Law, Economics, Korean-History) | 20개 |
| STEM (Computer-Science, Biology, Chemistry, Math) | 20개 |
| Applied Science (IT, Electrical, Environmental, Energy) | 20개 |
| Other (Health, Real-Estate, Public-Safety) | 15개 |
| **합계** | **75개** |

영어 벤치마크를 번역한 것이 아닌, 한국어로 원본 제작된 데이터셋으로  
한국어 LLM의 언어·문화적 특수성을 평가하는 데 적합합니다.

---

## 파이프라인 구조

```
[1] load_dataset.py     KMMLU에서 75개 질문 로드 (15개 subset × 5문제)
        ↓
[2] generate.py         모델 3개 × 전략 3가지로 응답 생성 (총 675개)
        ↓
[3] judge.py            Single Judge 채점 (675건)
        ↓
[4] evaluate.py         결과 집계 및 요약 출력
        ↓
[5] visualize.py        히트맵 / 승률 막대 / 박스플롯 저장
```

### FastChat 파일과 대응 관계

| FastChat | 본 프로젝트 | 변경사항 |
|---|---|---|
| `download_mt_bench_pregenerated.py` | `load_dataset.py` | KMMLU 한국어 데이터로 교체 |
| `gen_model_answer.py` | `generate.py` | Groq 무료 API로 교체, 프롬프트 전략 추가 |
| `judge_prompts.jsonl` | `judge_prompts.jsonl` | 한국어 Judge 프롬프트 추가 |
| `gen_judgment.py` | `judge.py` | Single + Pairwise 둘 다 구현 |
| `show_result.py` | `evaluate.py` | 카테고리별 분석 추가 |
| *(없음)* | `visualize.py` | 신규 추가 |
| *(없음)* | `llm_config.py` | 설정값 분리 |
| *(없음)* | `run.py` | 전체 파이프라인 단일 진입점 |

---

## Judge 방식

### 1. Single Grading
응답 1개를 Judge LLM이 1~10점으로 채점합니다.

```
[질문] + [응답] → Judge → "Rating: [[7]]"
```

### 2. Pairwise Comparison
같은 질문에 대한 두 모델의 응답을 Judge가 직접 비교합니다.

```
[질문] + [모델A 응답] + [모델B 응답] → Judge → "[[A]]" / "[[B]]" / "[[C]]"
```

### Judge 프롬프트 (`judge_prompts.jsonl`)

| 이름 | 방식 | 언어 |
|---|---|---|
| `single-v1` | Single | 영어 |
| `single-ko-v1` | Single | 한국어 |
| `pair-v2` | Pairwise | 영어 |
| `pair-ko-v1` | Pairwise | 한국어 |

---

## 실험 설계

### 비교 모델
| 모델 | Provider | 비용 |
|---|---|---|
| LLaMA 4 Scout 17B | Groq | 무료 |
| LLaMA 3.1 8B | Groq | 무료 |
| LLaMA 3.3 70B | Groq | 무료 |

### 프롬프트 전략
| 전략 | 설명 |
|---|---|
| Zero-shot | 질문만 그대로 입력 |
| Few-shot | 예시 1개 포함 후 질문 |
| CoT | "단계적으로 생각해봅시다" 유도 |

---

## 디렉터리 구조

```
LLM-evaluation/
├── llm_config.py        설정값 (모델, 경로, 전략 등)
├── load_dataset.py      KMMLU 데이터 로드 → questions.json 생성
├── generate.py          모델별 응답 생성
├── judge.py             Single + Pairwise Judge 채점
├── evaluate.py          결과 집계 및 요약
├── visualize.py         시각화 (히트맵, 승률, 박스플롯)
├── run.py               전체 파이프라인 실행
└── data/
    ├── questions.json        로드된 질문 (75개)
    ├── judge_prompts.jsonl   Judge 프롬프트 모음
    └── results/
        ├── responses.json        모델별 응답 (675개)
        ├── scores.json           Single Judge 채점 결과
        ├── pairwise.json         Pairwise Judge 결과
        ├── heatmap_single.png
        ├── pairwise_winrate.png
        └── strategy_boxplot.png
```

---

## 빠른 시작

### 1. 설치

```bash
pip install groq datasets pandas matplotlib numpy
```

### 2. API 키 등록

```bash
# macOS / Linux
export GROQ_API_KEY="gsk_..."

# Windows
set GROQ_API_KEY=gsk_...
```

API 키 발급: [console.groq.com](https://console.groq.com)

### 3. 실행

```bash
# 데이터 로드 (최초 1회)
python load_dataset.py

# 전체 실험 실행
python run.py
```

---

## 실험 결과 예시

### Single Judge — 모델 × 전략별 평균 점수

| 모델 | zero_shot | few_shot | cot | 평균 |
|---|---|---|---|---|
| llama-4-scout | 7.2 | 7.5 | 7.8 | **7.5** |
| llama-3.3-70b | 6.8 | 7.0 | 7.1 | **7.0** |
| llama-3.1-8b | 6.1 | 6.4 | 6.6 | **6.4** |

### Pairwise Judge — 모델별 승률

| 모델 | 승 | 패 | 무 | 승률 |
|---|---|---|---|---|
| llama-4-scout | 38 | 12 | 10 | **63%** |
| llama-3.3-70b | 28 | 22 | 10 | **47%** |
| llama-3.1-8b | 14 | 36 | 10 | **23%** |

> 실제 실험 결과 이미지는 `data/results/` 폴더 참고

---

## 주요 발견 (예시)

- **CoT 전략**이 전 모델에서 일관되게 점수를 높임 (평균 +0.4점)
- **Law 카테고리**에서 모델 간 점수 격차가 가장 크게 나타남
- **Pairwise와 Single 결과**가 모델 순위에서 높은 일치도를 보임
- **Verbosity Bias** 현상 관찰: Judge가 긴 응답을 선호하는 경향

---

## LLM Judge의 한계

논문(Zheng et al., NeurIPS 2023)에서 지적한 한계를 실험에서 직접 확인:

- **Position Bias**: Pairwise에서 먼저 제시된 응답을 선호하는 경향
- **Verbosity Bias**: 짧고 정확한 답변보다 길고 장황한 답변에 높은 점수
- **Self-enhancement Bias**: Judge 모델 자신의 스타일과 유사한 응답 선호

---

## 향후 개선 방향

- [ ] HAE-RAE Bench (한국어 문화·어휘) 추가
- [ ] Multi-judge 앙상블로 채점 편향 완화
- [ ] Reference-guided Judge 적용 (KMMLU 정답 활용)
- [ ] Agent Benchmark (tool use, 멀티턴) 평가로 확장

---

## 참고

- 논문: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (NeurIPS 2023)](https://arxiv.org/pdf/2306.05685)
- 원본 코드: [FastChat / lm-sys](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
- 데이터셋: [KMMLU (HAERAE-HUB)](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)

---

## 사용 기술

`Python` `Groq API` `HuggingFace Datasets` `Pandas` `Matplotlib`
