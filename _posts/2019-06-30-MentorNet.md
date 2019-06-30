---
title: "MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels"
date: 2019-06-30 08:26:28 -0400
categories: [Deep_Learning]
tags: [Deep_Learning,HS]
---

CL, SPL, SPCL 은 커리큘럼 러닝의 아이디어를 기반으로 학습을 시키되 어떤 데이터 `부터` 학습을 시킬지에 대한 문제에 대해 고민하고, 방법을 제시함으로써 모델의 수렴 속도를 개선하고 성능을 높이는 방법론이었습니다.

기존의 이러한 방법론들은 논문 별로 함수 $G$ 가 제시되어 왔습니다. 이를 Predefined Curriculum 이라는 이름으로 불렸었는데, 높은 성과를 보여주었지만, 모델을 직접 학습하는 모델의 피드백(학생의 피드백)을 충분히 받기 힘들었고, 데이터에 기인(data-driven)하지 않기 때문에, 함수 자체의 한계가 존재하게 되었습니다.

이러한 부분을 극복하기 위해서, 학습시키려는 모델(StudentNet)과 함께 학습되는(jointly trained) 또다른 모델(MentorNet)을 제안하는 [논문](https://arxiv.org/pdf/1712.05055.pdf) MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels 2017년에 나왔고, 이에 대해 이야기해보려 합니다.  

______________________


## 서론

해당 논문의 더 나은 이해를 위해서, 기존의 [방법론](https://hskimim.github.io/Self-Paced-Learning/)에 대해 훑고 오시는 것을 적극 추천드립니다!!

우선 해당 논문의 흥미로운 점은 커리큘럼 학습의 아이디어를 어디에 쓰려고 했느냐 에서부터 시작합니다. 이전 연구들이 이야기하던 CL 의 목적은 쉬운 샘플부터 시작해, 점차 어려운 샘플들로 가보자!! 였습니다. 이를 통해서 데이터 샘플의 가중치 $v_{i}$ 를 학습 iteration에 따라 변화해나갔죠.

하지만 이번 논문의 경우에는 그 목적이 신박하게 다릅니다. 바로 잘못 매겨진 라벨들에 (corrupted samples) 대해 적은 가중치를 주자!! 입니다. 잘못 매겨진 라벨들에 대한 문제와 CL이 어떠한 관계가 있는지에 대해 좀 더 이야기를 해보자면 이와 같습니다. 딥러닝 아키텍쳐는 깊을수록 성능이 좋은 경향성이 있습니다. 하지만, overfitting의 경향성 또한 깊어가죠. 이에 따라서 corrupted labels에 대해, 그대로 학습하여서, 아무리 좋은 모델이라고 해도 noisy label에 대해 속수무책으로 모든 것을 기억해버리고 맙니다.

 이러한 문제를 (overfitting) 해결하기 위해 regularization, dropout 과 같은 정규화를 시도하지만, 주된 문제는 바로 "corrupted label을 어떻게 정의내리며 이러한 데이터에 overfit 하지 않기 위한 방법이 어떤 것이 있을까" 입니다. 우선, corrputed label이 무엇인지에 대해 알아내야겠네요! corrputed label의 경우, 모델이 굉장히 혼란을 겪을 것이기 때문에, (content와 label이 misclassified 되어 있을 것이기 때문) loss가 크게 할당될 것입니다. 그러면 이 데이터에 대해서는 사알짝만 학습하도록 가중치를 줄여준다면? 특정 데이터(corrputed sample)에 대해서 overfit 효과를 줄일 수 있게 될 것입니다. 눈치채셨겠지만, 처음에 시도하는 단계가 바로 hard sample 을 찾는 과정과 유사하고, 두번째 시도하는 단계가 hard sample 에 대해 $v_{i}$를 줄이는 과정과 유사합니다. 이에 따라, 저희는 CL을 통해 잘못 정의된 라벨 데이터에 대해 과적합을 줄이면서 학습이 가능케 됩니다!! (와!!)

## 학습

모델을 학습시키는 방법은 크게 predefined curriculum 에 근접하게 학습시키는 방법과, 데이터 자체로 학습시키는 방법 두 가지가 있습니다. 하나씩 살펴보겠습니다.

predefined curriculum 방식은 아래 식의 $G$ 의 형태를 이미하는 상태를 의미합니다.
$$argmin_{\theta} \sum_{(x_{i},y_{i}) \in \mathbb(D)} g_{m} (z_{i};\theta)l_{i} + G(g_{m}(z_{i};\theta);\lambda)$$

해당 논문에서 사용한 함수는 이전 포스팅 [SPCL](https://hskimim.github.io/Self-Paced-Curriculum-Learning/) 의 마지막 부분에 나왔던 Mixture scheme function 과 일치하는 부분입니다.
$$G(v;\lambda) = \sum_{i=1}^{n} \ \frac{1}{2}\lambda_{2}v_{i}^2 - (\lambda_{1} + \lambda_{2})v_{i}$$$$g_m(z_{i},\theta) = \\ \mathbb{I}(l_{i},\lambda_{1}) \ \  \text{if} \ \ \lambda_{2} = 0 \\ min(max(0,1-\frac{l_{i}-\lambda_{1}}{\lambda_{2}}),1) \ \  \text{if} \ \ \lambda_{2} \ne 0$$

$\lambda_{1},\lambda_{2}$ 은 하이퍼 파라미터입니다. 모델의 손실 함수에 따라서 $l_{i}$ 이 도출되며, 이에 따라, predetermined curriculum function 이 $\hat{v_{i}}$ 를 반환합니다. 해당 방법론에서 MentorNet 은 지도학습의 형태로 $\hat{v_{i}}$ 을 라벨로 이를 학습하게 됩니다.

두 번째 방법론의 경우는 보다 간단합니다. 이번에는 저희가 기존에 알고 있던 지도 학습의 양상과 동일한데, $\hat{D} = \{(\pi(x_{i},y_{i},w),\hat{v_{i}})\}$ , $\|\hat{D}\| < \|D\|$ (`^` 은 optimal value 를 의미합니다.)  
여기서 $\hat{D}$ 은 well-defined label sample 입니다. 잘 되어 있는 경우 1, 아닌 경우 0으로 optimal value 가 할당되어서, MentorNet 이 뱉어내는 $v_{i}$ 가 cross entropy loss 를 계산하면서 학습을 진행하게 됩니다. 실제 실험에서는 CIFAR-10 의 라벨 데이터 5,000개를 MentorNet 을 학습시키는 데에 사용했다고 합니다.

MentorNet 의 학습뿐만 아니라, StudentNet을 학습시키는 프로세스가 존재하는데, 바로 burn-in period 를 할당하는 것입니다. 해당 period에서 MentorNet이 뱉어내는 값은 학습된 값이 아닙니다. $ v_{i} ∼ Bernoulli(p)$ 즉, 랜덤한 확률로 데이터 샘플을 학습하지 않는 것과 같은 것으로, $p\%$ 로 dropout 하는 것과 같은 맥락이 됩니다.

## 아키텍쳐

이제는 MentorNet의 아키텍쳐에 대해서 알아보도록 하겠습니다. 모델 자체는 크게 복잡하지 않습니다. 전형적인 MLP 형식을 사용하였고, output 으로는 확률값의 형태로 $v$ 를 뱉어내는 probabilistic sampling layer가 존재합니다.

<img src = "/images/post_img/markdown-img-paste-20190701011133332.png">

모델의 인풋은 그게 4가지가 존재합니다.

- $l$ : StudentNet 이 반환한 loss
- $l - l_{pt}$ : loss difference
  - $l_{pt}$ : $p-th$ percentile loss 를 지수 이동 평균 취한 값입니다.
    - 이전 iteration 이 될 수록, 그 영향력이 decay 되는 것이 EWMA 의 특징인데, 해당 논문에서 decay factor는 0.95를 주었고, percentile $p$는 cross-validation 으로 계산하였다고 합니다.
- label : True label
- training epoch percentage : [0,99] (%)

## 알고리즘

이전에 CL을 보였던 논문들은 $w,v$를 번갈아가면서 하나를 고정하고 나머지 하나를 최적화하는 방식으로 학습을 진행하였습니다. 하지만 그러한 방식은 deep-CNN을 사용할 경우 아래와 같은 문제점이 발생하게 됩니다.

1. 두 가지 변수를 따로따로 학습시키면, 시간도 오래걸리며, 메모리 할당이 너무 크다.
2. GPU를 통한 분산처리가 효율적이기 힘들다.

<img src = "/images/post_img/markdown-img-paste-20190701015149845.png">

이에 따라, SPADE라는 알고리즘을 제안하게 되는데, 쉽게 말씀드리자면, mini-batch 를 input 으로 하여서, StudentNet과 MentorNet을 동시에 학습시킨다 의 맥락이 됩니다. 9번째 줄이 의미하는 것은 predefined curriculum approach 의 경우, MentorNet의 gradient decsent 가 되고, 11번째 줄의 경우, data-driven curriculum approach 의 경우, MentorNet의 gradient decsent가 됩니다. 마지막으로 12번째 줄의 경우, StudentNet의 gradient descent 가 되는데, 이때 input 이 같은 iteration $t$에서 MentorNet 이 반환한 $v_{t}$가 됩니다.

## 실험

<img src = "/images/post_img/markdown-img-paste-20190701015403620.png">

위의 표에서 x축이 의미하는 것은 각 데이터와 사용된 StudentNet의 아키텍쳐, 그리고 의도적으로 생성한 noisy label의 비율을 의미합니다. 대부분의 경우에서, Data-driven MentorNet이 우위를 차지하고 있는 것을 확인할 수 있습니다.

<img src = "/images/post_img/markdown-img-paste-20190701015525429.png">

어탠션 메커니즘은 사용하면 확률값의 형태를 통해, DNN이 어떠한 경로와 의사결정을 통해, 결론이 도출되었는지를 확인할 수 있는 것과 마찬가지로 MentorNet이 반환하는 확률값의 형태인 $v$ 를 통해서도 모델을 해석할 수 있게 됩니다. 모델 epoch 를 21(초기) 번 정도 돌렸을 때에는, loss 크기에 따른, weight $v$의 변화가 눈에 띄게 나타나지 않습니다. 하지만 epoch를 더 돌려 76번에 다다랐을 때에는, loss가 증가하면서, weight $v$가 감소하는 양상을 위의 그래프를 통해 확인할 수 있게 됩니다.  

## 결론

세상에는 라벨이 없는 경우가 많습니다. 이에 따라서, 굉장히 나이브한 방식으로 라벨링을 하는 경우가 존재하는데 이러한 경우에, 믿을 수 없는 성능이나 Validation Performance 의 향상을 위해 Data driven CL 은 매우 좋은 학습 방법론이 될 것 같습니다. 읽어주셔서 감사합니다!!
