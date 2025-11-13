### 1. 데이터사이언스 공통

- 평가 항목: Python 기반 데이터 처리 언어 활용 능력
- 수행 내용:
  - 데이터 처리부터 실험, 분석까지 전 과정을 파이썬(`Python`)으로 수행
  - `pandas` 모듈을 사용해 구분자(`::` or `;`)가 제각각인 원본 텍스트 파일들을 표준 DataFrame으로 로드, `re`(정규표현식) 모듈을 이용해 `categories` 변수의 오기입(예: "childrens")이나 특수문자 등을 일괄 정제하는 데이터 표준화를 구현
  - `numpy`를 활용해 사용자의 모든 시청/구매 기록(`Train Set`)을 분석, "이 사용자가 얼마나 다양한 주제를 보는지" (`Gini Impurity`)와 "얼마나 새로운 것을 탐색하는지" (`Novelty`) 같은 개인화 프로필을 사전 계산
  - `sklearn`의 `train_test_split`을 기본 분할 로직으로 활용하되, `config.yaml`의 `time_aware: true` 설정 시 `timestamp`를 기준으로 데이터를 시간 순서대로 정렬하여 `Train/Test`를 나누는 `split_data` 함수를 별도 구현.
- 관련 코드:
  - `src/data_loader.py` (데이터 로딩 및 전처리)
  - `src/user_profiling.py` (사용자 프로필 계산)

---

### 2. 데이터 활용 및 분석

- 평가 항목: 머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부
- 수행 내용:
  - 추천 시스템 오픈소스 라이브러리인 `cornac` 을 활용해 `WMF`, `EASE`, `VAECF`, `NeuMF` 와 같은 4가지 표준 추천 모델을 이를 학습하고 평가하는 `run_model` 구현.
  - 베이스라인 모델이 뽑은 추천 목록을 입력받아, `관련성`과 `다양성`의 균형을 다시 맞추는 `Reranker` 클래스(`src/reranker.py`)를 구현
  - `Reranker`는 `run.py` 실행 시 미리 계산된 사용자 프로필(`Gini`, `Novelty`)을 불러옴. 그리고 $\lambda$(람다) 가중치에 따라, "사용자가 새로운 걸 좋아하면(Novelty 높음)" 숨겨진 아이템(A-DIV)에, "사용자가 넓게 들으면(Gini 높음)" 다양한 주제(I-DIV)에 가산점을 주는 식의 룰로 동작.
  - `run_sensitivity.py` 스크립트를 별도 구현하여, 람다($\lambda$) 가중치를 0.1~0.9까지 바꿔가며 실행하는 민감도 분석 수행
- 실행 방법:

  ```bash
  # config 파일 내 기입된 파라미터로 최종 실험 실행
  python run.py --config configs/movielens_1m.yaml --step rerank

  # 정확도와 다양성 간 트레이드오프 람다(Lambda) 민감도 분석 실행
  python run_sensitivity.py --config configs/movielens_1m.yaml
  ```

- 관련 코드:
  - `run.py` (메인 실행)
  - `src/reranker.py` (리랭킹 알고리즘)
  - `config/movielens_1m.yaml` (실험 관련 모든 파라미터)

---

### 3️⃣ 데이터 시각화

- 평가 항목: Python, R 등을 활용한 데이터 시각화 능력
- 수행 내용:
  - `run.py`가 생성한 결과 파일(`.pkl`, `.csv`)을 Jupyter Notebook (`notebooks/`)에서 불러와 분석하고 시각화함
  - `matplotlib`, `seaborn` 라이브러리를 사용해 다수의 실험 결과를 쉽게 이해할 수 있는 그래프로 시각화함
  - `notebooks/2_ANOVA_Analysis.ipynb`에서, 사용자 그룹(예: 취향 넓은 그룹 vs 좁은 그룹) 간에 베이스라인 모델의 성능이 통계적으로 차이 나는지(`scipy`, `statsmodels`)를 검증하고, `seaborn` 막대그래프로 시각화
  - `notebooks/3_Plot_Sensitivity.ipynb`에서, 람다($\lambda$) 값이 변할 때 '정확도(NDCG)'와 '다양성(Novelty)'이 어떻게 변하는지 `seaborn` 라인 플롯으로 시각화
- 관련 코드:
  - `notebooks/2_ANOVA_Analysis.ipynb` (그룹별 성능 차이 분석 및 시각화)
  - `notebooks/3_Plot_Sensitivity.ipynb` (람다 값 민감도 분석 시각화)

![Example_result](./results/movielens_1m/20251112_movielens_1m_VAECF_sensitivity_plot.png
)