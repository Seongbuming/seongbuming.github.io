---
layout: post
title: LLM의 Explainability
subtitle: 평가 방법과 과제
categories: Memo
tags: [Large Language Model]
use_math: true
---

Explayinability는 LLM의 의사결정 과정을 인간이 이해할 수 있는 형태로 설명하는 능력을 의미한다. 모델의 신뢰성과 투명성을 높이는 데 중요한 역할을 한다. Explainability는 크게 두 가지 패러다임으로 나뉜다: (1) 전통적인 fine-tuning 패러다임과 (2) prompting 패러다임이다. Explainability는 모델의 신뢰성을 높이고, 인간이 모델의 예측 결과를 더 잘 이해할 수 있게 하며, 궁극적으로는 모델의 오류를 더 잘 탐지하고 수정할 수 있도록 도와준다.

## 1. 전통적인 fine-tuning 패러다임의 explainability

이 패러다임에서는 pre-trained 모델을 특정 downstream task에 맞게 fine-tuning한다. 주요 설명 방법은 다음과 같다:

1. **Feature attribution-based explanation**: 입력 특성의 중요도를 측정하여 모델의 예측을 설명한다. 대표적인 방법으로는 LIME (Local Interpretable Model-agnostic Explanations)와 SHAP (SHapley Additive exPlanations) 등이 있다. 이 방법들은 각 입력 특징이 모델의 출력에 얼마나 기여있는지를 정량적으로 보여준다.
2. **Attention-based explanation**: 모델의 attention 메커니즘을 분석하여 중요한 입력 부분을 식별한다. Attention 메커니즘은 특히 Transformer 기반 모델에서 많이 사용되며, 입력 시퀀스의 어느 부분에 모델이 주목했는지를 시각적으로 나타낼 수 있다.
3. **Example-based explanation**: 모델의 예측을 설명하기 위해 유사한 훈련 데이터 예시를 제공한다. 예를 들어, 모델이 특정 입력에 대해 예측을 내릴 때, 유사한 입력을 가진 훈련 데이터의 예를 제공함으로써 설명을 제공할 수 있다.
4. **Natural language explanation**: 모델이 직접 자연어로 예측 이유를 설명한다. 이는 모델이 왜 특정 예측을 했는지를 텍스트 형식으로 설명하는 방식으로, 사용자가 더 쉽게 이해할 수 있게 한다.

## 2. Prompting 패러다임의 explainability

이 패러다임에서는 모델에 prompt를 제공하여 zero-shot 또는 few-shot learning을 수행한다. 주요 설명 방법은 다음과 같다:

1. **In-context learning explanation**: 모델이 주어진 맥락에서 어떻게 학습하고 추론하는지 분석한다. 모델이 몇 가지 예시만을 가지고도 새로운 태스크를 수행할 수 있는 능력을 의미한다. 이 과정을 통해 모델이 어떤 식으로 정보를 활용하는지를 설명할 수 있다.
2. **Chain-of-thought prompting explanation**: 모델의 단계별 추론 과정을 분석한다. 이는 모델이 복잡한 문제를 해결할 때 거치는 중간 단계를 명시적으로 보여주어, 예측이 어떻게 도출되었는지를 이해할 수 있게 한다.
3. **Representation engineering**: 모델의 내부 표현을 분석하고 조작하여 모델의 행동을 이해한다. 모델의 내부 상태나 중간 결과를 분석하여, 입력이 어떻게 변환되고 있는지를 이해하는 방법이다.

## 3. LLM의 explainability 평가 방법

Explainability의 평가는 크게 두 가지 측면에서 이루어진다: plausibility(타당성)와 faithfulness(충실도)이다.

### 3.1. Plausibility 평가

Plausibility는 설명이 인간에게 얼마나 타당하게 보이는지를 평가한다.

1. **Intersection-Over-Union (IOU)**: $\text{IOU} = \frac{\text{인간 주석 토큰} \cap \text{모델 생성 토큰}}{\text{인간 주석 토큰} \cup \text{모델 생성 토큰}}$
2. **Precision**: $\text{Precision} = \frac{\text{정확히 식별된 중요 토큰 수}}{\text{모델이 식별한 총 토큰 수}}$
3. **Recall**: $\text{Recall} = \frac{\text{정확히 식별된 중요 토큰 수}}{\text{실제 중요 토큰의 총 수}}$
4. **F1 score**: $\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **Area Under the Precision Recall Curve (AUPRC)**: 다양한 임계값에 대한 Precision-Recall 곡선의 아래 영역을 계산한다.

### 3.2. Faithfulness 평가

Faithfulness는 설명이 모델의 실제 의사결정 과정을 얼마나 정확하게 반영하는지를 평가한다.

1. **Comprehensiveness (COMP)**: 원본 예측 확률에서 중요 토큰을 제거한 후의 예측 확률을 뺀다. $\text{COMP} = P(y \vert x) - P(y \vert x\ \backslash\ x_{important})$
2. **Sufficiency (SUFF)**: 원본 예측 확률에서 중요 토큰만으로의 예측 확률을 뺀다. $\text{SUFF} = P(y \vert x) - P(y \vert x_{important})$
3. **Decision Flip - Fraction of Tokens (DFFOT)**: 결정이 뒤집힐 때까지 제거된 토큰의 평균 비율을 계산한다.
4. **Decision Flip - Most Informative Token (TFMIT)**: 가장 영향력 있는 토큰 제거 시 결정이 뒤집히는 비율을 계산한다.

### 3.3. Prompting 패러다임에 특화도니 평가

1. **Simulation generality**: 설명이 도움이 되는 다양한 counterfactual 입력의 수를 측정한다.
2. **Simulation precision**: $\text{Simulation Precision} = \frac{\text{인간 예측과 일치하는 모델 출력 수}}{\text{총 시뮬레이션된 counterfactual 수}}$

## 4. 평가 시 고려사항과 과제

1. **Ground truth explanation의 부재**: 많은 경우 정답 설명이 없어서 평가가 어렵다.
2. **인간 평가와 자동화된 평가의 구현**: 인간의 주관적 판단과 객관적인 지표 사이의 균형을 맞추는 것이 중요하다.
3. **다양한 태스크와 모델에 대한 벤치마크**: 다양한 상황에서의 일관된 평가를 위해 포괄적인 벤치마크가 필요하다.
4. **설명의 일관성 및 안정성**: 유사한 입력에 대해 일관된 설명을 제공하는지 평가해야 한다.

## 5. 향후 연구 방향

1. **표준화된 평가 지표 개발**: 더욱 엄격하고 일관된 평가를 위한 지표가 필요하다.
2. **Faithfulness와 plausibility 동시 향상**: 두 가지 측면을 모두 개선하는 방법론 연구가 필요하다.
3. **LLM 특화 평가 방법**: 대규모 언어 모델의 특성을 고려한 새로운 평가 방법 개발이 필요하다.
4. **설명의 인과관계 분석**: 모델의 의사결정과 설명 사이의 인과관계를 더 깊이 이해하는 연구가 필요하다.
5. **다국어 및 다중 모달 설명**: 다양한 언어와 모달리티에 걸친 설명 및 평가 방법 연구가 필요하다.

결론적으로, LLM의 explainability는 모델의 투명성과 신뢰성을 높인다. 다양한 평가 방법을 통해 설명의 질을 개선하고, 이를 통해 AI 시스템의 책임 있는 발전을 도모할 수 있다. 향후 연구에서는 더욱 정교하고 표준화된 평가 방법과 실제 응용에 적용 가능한 설명 기술의 개발이 필요할 것이다.
