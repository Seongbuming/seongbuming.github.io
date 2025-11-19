---
layout: post
title: AI 시스템의 Observability
subtitle: 프로덕션 환경에서의 LLM 모니터링과 신뢰성 확보
categories: AI Engineering
tags: [LLM, Observability, Production, Monitoring]
use_math: true
---

프로덕션 환경에서 대규모 언어 모델(Large Language Model, LLM)을 운영하는 것은 전통적인 소프트웨어 시스템을 운영하는 것과는 다른 도전 과제를 제시한다. LLM은 확률적이고, 그 내부 작동 방식은 불투명하며, 수백만 개의 예측 불가능한 응답을 생성할 수 있다. 이러한 특성은 시스템의 관찰 가능성(observability)을 확보하는 것을 필수적이면서도 어려운 과제로 만든다. Observability는 시스템을 모니터링하는 것을 넘어서, 시스템의 내부 상태를 외부 출력을 통해 이해하고, 문제가 발생했을 때 그 원인을 파악하며, 시스템이 의도한 대로 작동하고 있는지를 지속적으로 검증하는 능력을 의미한다.

## 전통적 시스템과의 차이

전통적인 소프트웨어 시스템에서는 로그, 메트릭, 트레이스라는 세 가지 기둥(three pillars of observability)을 통해 시스템의 상태를 파악한다. 그러나 AI 시스템, 특히 LLM 기반 시스템은 이러한 전통적 관찰 가능성 프레임워크만으로는 충분하지 않다. OpenTelemetry 커뮤니티가 2024년 6월에 발표한 가이드에 따르면, LLM 애플리케이션은 외부 API 호출을 통해 접근되는 경우가 많으며, RAG(Retrieval-Augmented Generation) 기반 애플리케이션에서는 LLM 사용 전후의 이벤트 시퀀스를 추적하는 것이 중요하다[^1].

전통적인 ML 모델과 비교해도 LLM의 특성은 독특하다. 전통적인 ML 모델은 입력과 출력 간의 관계가 상대적으로 명확하고, 예측 결과를 정량적으로 평가할 수 있는 ground truth가 존재하는 경우가 많다. 반면 LLM은 동일한 입력에 대해서도 다양한 출력을 생성할 수 있으며, 그 출력의 품질을 평가하는 것 자체가 복잡한 문제다. Datadog의 2024년 보고서에서 지적한 바와 같이, LLM의 대규모 모델 크기, 복잡한 아키텍처, 그리고 비결정적(non-deterministic) 출력은 전통적인 ML 애플리케이션에 비해 프로덕션 운영을 더욱 어렵게 만든다[^2].

더 나아가, LLM 시스템은 모델 자체의 행동뿐만 아니라 프롬프트 엔지니어링, 컨텍스트 관리, 외부 지식 베이스와의 통합 등 여러 레이어가 복잡하게 얽혀 있다. 이러한 복잡성은 단순한 메트릭 추적을 넘어서는 정교한 관찰 가능성 전략을 요구한다.

## LLM Observability의 메트릭

LLM 시스템의 관찰 가능성을 확보하기 위해서는 여러 차원의 메트릭을 동시에 추적해야 한다. 이러한 메트릭들은 크게 성능, 품질, 비용, 그리고 보안의 네 가지 카테고리로 분류할 수 있다.

성능 측면에서는 레이턴시(latency)와 처리량(throughput)이 가장 기본적인 지표다. LLM의 응답 시간은 네트워크 지연과 모델 자체의 생성 시간을 모두 포함하며, 이는 사용자 경험에 직접적인 영향을 미친다. OpenTelemetry 가이드는 요청 지속 시간(Request Duration)을 핵심 신호(key signal)로 제시하며, 이는 요청이 처리되고 LLM으로부터 응답을 받는 데 걸리는 시간으로, 성능과 신뢰성에 대한 통찰을 제공한다고 설명한다[^1].

품질 메트릭은 LLM observability의 가장 도전적인 영역이다. 전통적인 NLP 메트릭인 BLEU, ROUGE, METEOR는 기계적인 품질 기준선을 제공하지만, 사용자가 즉시 발견하는 의미론적 불일치(semantic inconsistencies)를 놓치는 경우가 많다[^3]. 이를 보완하기 위해 최근에는 embedding 기반 기법이 주목받고 있다. Embedding 분포는 모델이 패턴에서 벗어난 응답을 생성하기 시작할 때를 드러내는 미묘한 의미론적 단서를 인코딩한다. 클러스터링 방법, 이상 탐지(anomaly detection), t-SNE와 같은 시각화 기법을 활용하여 고차원 공간에서 이러한 변화를 조기에 식별할 수 있다[^3].

비용 관리는 프로덕션 LLM 시스템에서 간과할 수 없는 요소다. 토큰 사용량과 API 호출 빈도를 추적하는 것은 예산 관리와 비용 최적화 전략에 필수적이다. OpenTelemetry는 시간 경과에 따른 총 비용과 소비된 토큰을 추적하는 것이 예산 책정과 비용 최적화 전략에 필수적이라고 강조한다. 이러한 메트릭을 모니터링하면 비효율적인 LLM 사용을 나타낼 수 있는 예기치 않은 증가에 대해 경고를 받을 수 있다[^1].

보안과 안전성 측면에서는 hallucination 감지, 프롬프트 인젝션(prompt injection) 방지, 민감 정보 유출 방지가 중요하다. Aporia의 플랫폼이 제공하는 것처럼, 실시간으로 모델의 프롬프트와 응답 임베딩을 모니터링하여 일관성 없는 부분을 식별하고, 비즈니스 운영이나 사용자 경험에 영향을 미치기 전에 대응하고 수정하는 것이 중요하다[^4].

## 드리프트 현상과 그 탐지

LLM 시스템에서 드리프트(drift)는 모델의 성능이나 행동이 시간이 지남에 따라 변화하는 현상을 의미한다. 이는 크게 데이터 드리프트와 모델 드리프트로 구분할 수 있다. 데이터 드리프트는 입력 데이터 분포의 변화를 의미하며, 모델 드리프트는 모델의 예측 성능이 저하되는 것을 말한다. LLM 환경에서는 데이터 드리프트(새로운 슬랭이나 주제)가 종종 모델 드리프트(관련성이 떨어지는 응답)를 야기한다[^5].

프롬프트 드리프트(prompt drift)는 LLM 특유의 현상이다. 이는 시간이 지남에 따라 프롬프트가 다른 응답을 산출하는 현상으로, 모델 변경, 모델 마이그레이션, 또는 추론 시점에서의 프롬프트 주입 데이터 변화에 의해 발생한다[^6]. 예를 들어, 전자상거래 플랫폼의 고객 서비스 챗봇이 "배송"이라는 단어를 물리적 패키지 배송으로 이해하도록 학습되었다가, 플랫폼이 디지털 제품 다운로드를 출시하면서 사용자들이 "배송"을 물리적 및 디지털 배송 모두를 지칭하기 시작하는 경우, 챗봇은 혼란을 겪게 된다[^7].

드리프트 탐지를 위한 실전 기법으로는 여러 가지가 있다. AWS는 임베딩 드리프트를 추적하는 방법으로 프롬프트 임베딩을 클러스터링하고, 새로운 쿼리가 새로운 클러스터를 형성하거나 클러스터 중심을 이동시키는지를 확인하는 방식을 제시했다[^8]. 각 클러스터의 데이터 비율이 기준선에서 새로운 스냅샷으로 크게 변화하거나 클러스터 중심이 이동하면, 이는 쿼리의 주제나 맥락이 진화하고 있음을 나타낸다.

임베딩 공간에서의 드리프트는 다음과 같이 정의할 수 있다. 시간 $t_1$과 $t_2$에서의 임베딩 분포를 각각 $P_{t_1}$과 $P_{t_2}$라고 할 때, 드리프트의 정도 $D$는 다음과 같이 측정할 수 있다:

$$D(P_{t_1}, P_{t_2}) = \text{KL}(P_{t_1} \parallel P_{t_2}) = \sum_{i} P_{t_1}(i) \log \frac{P_{t_1}(i)}{P_{t_2}(i)}$$

여기서 KL은 Kullback-Leibler divergence를 의미한다. 실무에서는 이를 직접 계산하는 대신, 클러스터링 기반 접근법이나 통계적 검정(Kolmogorov-Smirnov 테스트, Mann-Whitney U-테스트 등)을 활용한다[^9].

또한 카나리 프롬프트(canary prompts)라는 기법도 유용하다. 이는 변하지 않는 입력으로 설계되어 행동 드리프트를 표면화하는 고정된 프롬프트다. 모델이 이러한 프롬프트에 응답하는 방식의 변화는 추론 스타일, 톤, 또는 신뢰도의 변화를 나타낼 수 있으며, 이 모두는 사용자 경험에 영향을 미친다[^3].

## Hallucination 탐지와 대응

Hallucination은 LLM이 사실과 다르거나 오해를 불러일으키는 응답을 생성하는 현상으로, 프로덕션 시스템에서 가장 심각한 위험 중 하나다. LLM은 통계적 패턴을 기반으로 텍스트를 생성하지, 사실 검증을 수행하지 않는다. 맥락이 불충분하거나 모호할 때, 모델은 "공백을 채우기" 위해 만들어낸 내용으로 응답할 수 있다[^10].

실제 사례를 보면 그 위험성이 명확하다. 한 지원 챗봇이 존재하지 않는 환불 정책을 인용하여 법적 소송과 보상으로 이어진 사건이 있었다. 또한 변호사가 ChatGPT가 생성한 가짜 판례를 인용하여, 법원이 AI 생성 콘텐츠의 공개와 검증을 명령한 사례도 있다[^10].

Hallucination을 탐지하고 완화하기 위한 전략은 여러 층위에서 작동한다. 첫째, RAG(Retrieval-Augmented Generation) 패턴을 활용하여 검증된 문서에 응답을 근거하는 것이다. 이는 모델이 외부 지식 베이스에서 관련 정보를 검색한 후, 그 정보를 바탕으로 응답을 생성하도록 한다.

둘째, LLM-as-a-Judge 기법을 사용한다. 이는 보조 모델을 사용하여 출력 품질을 평가하는 방식이다. 예를 들어, GPT-4를 평가자로 사용하여 다른 LLM의 출력이 주어진 기준(정확성, 관련성, 안전성 등)을 충족하는지 판단할 수 있다.

셋째, Human-in-the-Loop 접근법으로, 고위험 상호작용에 대해 수동 검토를 통합한다. 의료, 법률, 금융 등 민감한 도메인에서는 이러한 인간 개입이 필수적이다.

마지막으로, 출력 가드레일(output guardrails)을 구현하여 주제에서 벗어나거나 안전하지 않은 응답을 감지하는 필터와 분류기를 설치한다. Superwise가 제공하는 것처럼, 사실 불일치, 깨진 인용, 콘텐츠 이상을 스캔하고, LLM이 만들어낸 내용을 감지하여 해당 응답을 플래그 지정하거나, 다른 경로로 전환하거나, 자동으로 차단할지 결정할 수 있다[^11].

## 실전 아키텍처와 구현

프로덕션 환경에서 LLM observability를 구현하기 위해서는 전체 스택을 아우르는 통합된 아키텍처가 필요하다. 현대적인 LLM 관찰 가능성 플랫폼은 여러 핵심 기능을 제공해야 한다.

먼저 추적(tracing) 기능이 있다. 복잡한 다단계 AI 에이전트 상호작용을 통한 워크플로우를 추적하는 것이다. Langfuse와 같은 플랫폼은 모델 및 프레임워크에 구애받지 않는 설계로, 추론 및 임베딩 검색부터 API 사용까지 LLM 애플리케이션의 전체 맥락을 캡처할 수 있도록 한다[^12].

다음으로 프롬프트 관리가 중요하다. 프롬프트의 버전 관리와 A/B 테스팅 기능을 제공하여, 프롬프트 변경이 출력 품질에 미치는 영향을 체계적으로 평가할 수 있어야 한다. PromptLayer와 같은 도구는 프롬프트 품질이 제품 성공을 직접적으로 결정하는 팀에게 이상적이며, 엄격한 프롬프트 관리와 실험이 필요한 조직에 적합하다[^13].

또한 비용 추적 및 최적화 기능도 필수적이다. 효율성 병목 지점을 식별하고 비용을 관리하기 위해 비용과 지연 시간을 추적하는 것이다. 토큰 사용량, API 호출 빈도, 그리고 이들이 실제 비즈니스 가치로 전환되는 비율을 모니터링해야 한다.

실제 구현 측면에서는 OpenTelemetry 기반의 접근법이 점점 더 표준으로 자리잡고 있다. Traceloop의 OpenLLMetry SDK는 LLM 관찰 가능성 데이터를 10개 이상의 다양한 도구로 전송할 수 있도록 하며, LLM 프로바이더나 LangChain, LlamaIndex와 같은 프레임워크에서 직접 추적을 추출하여 OTel 형식으로 게시한다[^14].

구체적인 구현 예시를 보면, 먼저 데이터 수집 레이어에서 프롬프트와 응답을 캡처한다. 이는 텍스트만을 저장하는 것이 아니라, 관련 메타데이터(타임스탬프, 사용자 ID, 세션 정보 등)와 함께 구조화된 형태로 저장해야 한다. 그런 다음 분석 레이어에서 이러한 데이터에 대해 다양한 메트릭을 계산하고, 드리프트나 이상 징후를 탐지한다. 마지막으로 액션 레이어에서 탐지된 문제에 대해 자동화된 대응(경고 발송, 트래픽 라우팅 변경, 폴백 모델로 전환 등)을 수행한다.

## 비용 효율성과 성능 최적화

프로덕션 LLM 시스템의 운영 비용은 빠르게 증가할 수 있다. 효과적인 observability 전략은 비용 최적화와 밀접하게 연결되어 있다. 토큰 사용 패턴을 분석하면 비효율적인 프롬프트를 식별할 수 있다. 예를 들어, 불필요하게 긴 컨텍스트를 반복적으로 전송하거나, 비슷한 쿼리에 대해 캐싱을 활용하지 않는 경우가 있다.

레이턴시 최적화도 중요한 관심사다. 사용자 경험에 직접적인 영향을 미치기 때문이다. Datadog의 LLM Observability가 제공하는 것처럼, 에이전트 워크플로우와 LLM 체인 전체에서 각 단계별로 입력, 출력, 지연 시간, 토큰 사용량, 오류를 추적하여 느린 응답 시간과 같은 프로덕션 병목 지점을 식별하고 문제를 해결할 수 있다[^2].

실무에서는 다음과 같은 최적화 전략이 효과적이다. 첫째, 프롬프트 최적화를 통해 동일한 품질의 응답을 더 적은 토큰으로 얻을 수 있다. 둘째, 응답 캐싱을 활용하여 동일하거나 유사한 쿼리에 대해 재계산을 피한다. 셋째, 모델 선택을 동적으로 조정하여, 간단한 쿼리에는 작은 모델을, 복잡한 쿼리에는 큰 모델을 사용한다. 넷째, 배치 처리를 활용하여 여러 요청을 함께 처리함으로써 처리량을 높인다.

Galileo AI가 제시하는 접근법처럼, 통합된 품질 및 드리프트 모니터링을 통해 출력 일관성, LLM hallucination 감지, 라이브 트래픽 전반의 의미론적 드리프트를 지속적으로 추적하는 것이 중요하다[^15]. 이는 운영 오버헤드를 증가시키지 않으면서도 신뢰성 관행을 확장할 수 있게 해준다.

## 보안과 컴플라이언스

LLM 시스템의 보안은 다층적인 접근을 요구한다. 프롬프트 인젝션 공격, 데이터 중독(data poisoning), 탈옥(jailbreaking), 정보 유출 공격을 실시간으로 감지해야 한다. Superwise가 제공하는 것처럼, 이러한 위협이 발생하는 순간 플래그를 지정하고, 근본 원인을 조사하며, 위협이 확대되기 전에 제어할 수 있어야 한다[^11].

민감한 정보 보호도 중요하다. PII(Personally Identifiable Information) 수정, 민감 데이터 스캐닝, 프롬프트 해킹에 대한 보호를 포함한 강력한 보안 기능을 제공해야 한다[^2]. 특히 의료, 금융, 법률 등 규제가 엄격한 산업에서는 이러한 보안 기능이 필수적이다.

컴플라이언스 측면에서는 EU AI Act와 같은 규제가 AI 시스템의 투명성과 책임성을 요구하고 있다. 조직은 더 이상 AI를 블랙박스로 취급할 수 없으며, 모델 행동과 의사 결정 프로세스에 대한 검증 가능한 기록이 필요하다[^13]. 이는 모든 입력, 출력, 중간 단계, 그리고 의사 결정에 영향을 미친 요소들을 추적하고 감사할 수 있는 시스템을 구축해야 함을 의미한다.

AgentOps 분류 체계(taxonomy)가 제안하는 것처럼, foundation model 기반 에이전트의 효과적인 관찰 가능성을 달성하기 위해서는 추적 가능성(traceability), 해석 가능성(interpretability), 제어 가능성(controllability), 디버그 가능성(debuggability)이라는 네 가지 주요 카테고리를 다뤄야 한다[^16]. 이러한 프레임워크는 안전하고 신뢰할 수 있으며 책임 있는 AI 에이전트를 개발하는 데 귀중한 가이드를 제공한다.

## 지속적 개선과 피드백 루프

효과적인 observability 시스템은 모니터링에 그치지 않고, 지속적인 개선을 위한 피드백 루프를 구축해야 한다. 프로덕션 환경에서 수집된 데이터는 모델을 개선하고, 프롬프트를 최적화하며, 시스템 아키텍처를 발전시키는 데 활용되어야 한다.

Langfuse가 제공하는 것처럼, 프로덕션 추적에서 직접 데이터셋을 생성하여 실제 시나리오에 대해 변경사항을 테스트하는 것이 중요하다. 또한 Playground를 사용하여 몇 분 안에 실험을 검증하고 비교하며, 프롬프트 조정, 모델 교체, 또는 파라미터 미세 조정을 테스트할 수 있다[^2].

2025년 LLMOps 보고서가 지적한 것처럼, 모니터링 없이 6개월 이상 변경되지 않은 모델은 새로운 데이터에 대해 오류율이 35% 증가했다[^5]. 이는 드리프트가 불가피하지만, 조기 탐지와 개입을 통해 "낡은" 모델이 사용자 경험을 해치는 것을 방지할 수 있음을 보여준다.

실무적으로는 다음과 같은 사이클을 구축하는 것이 효과적이다. 먼저 프로덕션에서 데이터를 수집하고, 이를 분석하여 패턴과 이슈를 식별한다. 그런 다음 가설을 수립하고 실험을 설계한다. 실험 결과를 평가하고, 성공적인 변경사항을 점진적으로 롤아웃한다. 이 과정에서 A/B 테스팅, 카나리 배포(canary deployment), 그리고 feature flag를 활용하여 위험을 최소화한다.

## 결론

LLM observability의 미래는 여러 방향으로 진화하고 있다. AI 보조 관찰 가능성(AI-assisted observability)이라는 개념이 등장하고 있는데, 이는 모델이 스스로를 모니터링하는 데 도움을 주는 방식이다. 평가 프레임워크와의 더 긴밀한 통합을 통한 자동화된 품질 테스트도 발전하고 있다. 또한 거버넌스 요구사항이 공고해짐에 따라 규제 컴플라이언스 기능이 표준이 되고 있다[^13].

시장 전망도 이러한 중요성을 반영한다. 2024년에 56억 달러로 평가된 대규모 언어 모델 시장은 2030년까지 거의 350억 달러로 폭발적으로 성장할 것으로 예상되며, 이는 연간 37%의 성장률이다. 동시에, AI 관찰 가능성 시장은 2023년 14억 달러에서 2033년 107억 달러로 급증하고 있어, 기업이 수조 달러 규모의 AI 시스템을 배포함에 따라 정교한 모니터링에 대한 긴급한 필요성을 반영하고 있다[^15].

결론적으로, AI 시스템의 observability는 기술적 요구사항을 넘어서 프로덕션 LLM 시스템의 성공을 위한 필수 요소가 되었다. LLM이 실험 단계에서 중요한 애플리케이션을 구동하는 프로덕션 시스템으로 이동함에 따라, 강력한 관찰 가능성은 신뢰할 수 있고 효율적이며 신뢰할 수 있는 AI를 구축하기 위한 기반이다.

효과적인 observability 전략은 성능 메트릭, 품질 평가, 비용 최적화, 보안 모니터링을 통합적으로 다뤄야 한다. 드리프트 탐지와 hallucination 방지는 지속적인 관심을 요구하며, 이를 위한 자동화된 시스템과 인간의 판단이 조화를 이뤄야 한다. 프로덕션 추적에서 수집된 데이터는 지속적인 개선의 원천이 되어, 모델 성능을 향상시키고 사용자 경험을 개선하는 선순환 구조를 만들어낸다.

앞으로 조직들은 자신의 특정 요구사항에 맞는 관찰 가능성 도구와 전략을 선택해야 한다. 인프라 최적화를 위한 AIBrix, 프롬프트 관리를 위한 PromptLayer, 심층 분석을 위한 Arize와 같이 특화된 솔루션들을 조합하는 것이 필요할 수 있다. 어떤 단일 플랫폼도 모든 것을 해결하지는 못하기 때문이다.

## 참고문헌

[^1]: Jain, I. (2024). "An Introduction to Observability for LLM-based applications using OpenTelemetry." OpenTelemetry Blog. https://opentelemetry.io/blog/2024/llm-observability/

[^2]: Datadog. (2024). "What Is LLM Observability & Monitoring?" Datadog Knowledge Center. https://www.datadoghq.com/knowledge-center/llm-observability/

[^3]: Galileo AI. (2025). "7 Strategies To Solve LLM Reliability Challenges at Scale." https://galileo.ai/blog/production-llm-monitoring-strategies

[^4]: Aporia. (2024). "LLM Observability for your production models." https://www.aporia.com/ml-observability/llm/

[^5]: Paul, R. (2025). "Handling LLM Model Drift in Production Monitoring, Retraining, and Continuous Learning." https://www.rohan-paul.com/p/ml-interview-q-series-handling-llm

[^6]: Greyling, C. (2023). "LLM Drift, Prompt Drift, Chaining & Cascading." Medium. https://cobusgreyling.medium.com/llm-drift-prompt-drift-chaining-cascading-fa8fbf67c0fd

[^7]: Siciliani, T. (2025). "Drift Detection in Large Language Models: A Practical Guide." Medium. https://medium.com/@tsiciliani/drift-detection-in-large-language-models-a-practical-guide-3f54d783792c

[^8]: Olaoye, A., et al. (2024). "Monitor embedding drift for LLMs deployed from Amazon SageMaker JumpStart." AWS Machine Learning Blog. https://aws.amazon.com/blogs/machine-learning/monitor-embedding-drift-for-llms-deployed-from-amazon-sagemaker-jumpstart/

[^9]: Nexla. (2024). "Data Drift in LLMs—Causes, Challenges, and Strategies." https://nexla.com/ai-infrastructure/data-drift/

[^10]: Paul, K. (2025). "LLM Monitoring: Detecting Drift, Hallucinations, and Failures." Medium. https://medium.com/@kuldeep.paul08/llm-monitoring-detecting-drift-hallucinations-and-failures-1028055c1d34

[^11]: Superwise. (2025). "LLM Monitoring | Track Drift, Risk & Hallucination." https://superwise.ai/llm-monitoring/

[^12]: Langfuse. (2024). "What is LLM Observability & Monitoring?" https://langfuse.com/faq/all/llm-observability

[^13]: PromptLayer. (2025). "Leading AI Visibility Optimization Platforms for LLM's Observability." https://blog.promptlayer.com/leading-ai-visibility-optimization-platforms-for-llms-observability/

[^14]: lakeFS. (2025). "LLM Observability Tools: 2025 Comparison." https://lakefs.io/blog/llm-observability-tools/

[^15]: Galileo AI. (2025). "Which LLM Observability Tools Prevent Failures in 2025?" https://galileo.ai/blog/best-llm-observability-tools-compared-for-2024

[^16]: AI Models. (2024). "A Taxonomy of AgentOps for Enabling Observability of Foundation Model based Agents." https://www.aimodels.fyi/papers/arxiv/agentops-enabling-observability-llm-agents
