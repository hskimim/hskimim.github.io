---
title: "Universal Language Model Fine-tuning for Text Classification"
date: 2019-09-07 08:26:28 -0400
categories: [NLP_paper]
tags: [NLP_paper,HS]
---

[gpt1](https://hskimim.github.io/Improving-Language-Understanding-by-Generative-Pre-training/) 과 [ELMO](https://hskimim.github.io/Deep-contextualized-word-representations/) 사이에 나온 논문, ULM-Fit 으로 잘 알려져 있는 [논문](https://arxiv.org/pdf/1801.06146.pdf)에 대해 간략하게 리뷰해보도록 하겠습니다.

해당 논문을 읽다가, 처음부터 막히는 부분이 있었는데요. 바로 Introduction의 transductive transfer 와 inductive transfer 에 대한 내용이었습니다. 해당 [링크](https://www.quora.com/What-is-the-difference-between-inductive-and-transductive-learning) 를 참고하여 나름대로 이해를 해본 결과를 정리해보겠습니다.

- transductive transfer :
  - 말 그대로 unlabeled data를 통해 학습의 정확도를 향상시키는 것.
  - 전형적인 sem-supervised Learning
  - 논문의 예시로 pretrained embedding 이 있다.

- Inductive transfer :
  - labeled data를 보고, unlabeled data를 예측하는 것
  - 전형적인 supervised learning

문제는 이 다음입니다. 이렇게 보면 ULM-Fit은 transductive transfer 이어야 하는데, 왜 inductive transfer라고 나와있을까요? (물론 저만 이해를 못하는 것일 수도 있습니다.)
이럴 때는 역시 짜맞추기를 해야겠죠? ULM-Fit에서 Universal 은 context representation 을 강화하기 위해서 universal scale로 단어를 임베딩한 단계를 의미합니다. 이 단계에서는 LM(Language Modeling) 을 통해서, 단어가 벡터로 표상됩니다.

그 다음에 fine-tuning이라는 bottleneck 과 유사한 projection layer가 들어가는데, 지도 학습을 통해 얇은 레이어의 파라미터가 학습되고 Fit 을 풀어쓴 것과 같이 for text classification 의 모델이 됩니다.

좀 더 자세히 보니 (짜맞추어 보니) pre-trained embedding과 다른 면이 보입니다. doc2vec, fasttext, glove과 같은 임베딩 벡터들은 unsupervised learning을 통해서 질높은 vector representation 을 구현하기 위한 기능을 하였습니다. ULM-Fit의 경우에는 언어 모델로 vector representation 을 강화시켰지만, 여기서 그치지 않고, fine-tuning이라는 프로세스를 거쳐 지도 학습의 형태로 information trasfer를 구현하였습니다.

해당 논문은 technical detail들이 다소 있다고 느껴졌던 논문이었습니다. 하나씩 살펴보면서 진행토록 하겠습니다.

### General-domain LM pretraining

언어 모델을 학습시킬 때에는 단어의 도메인에 구애받지 않고, 넓은 분야의 단어들을 학습시켰습니다. (28,595 preprocessed Wikipedia articles and 103 millions words.) 가장 오래 걸리는 작업이지만, performance를 향상시키고, convergence를 빠르게 해준다고 합니다.

### Target task LM fine-tuning
LM은 overfitting이 쉽게 된다는 단점이 있습니다. 이 문제를 해결하기 위해서 저자는 다양한 기술들을 가미했습니다.

#### Discriminative fine-tuning

아래의 공식은 gradient descent formula 입니다.

$$\theta_{t} = \theta_{t-1} - \lambda * J_{\theta^{'}}(\theta)$$

논문의 저자는 레이어 별로 다른 learning rate $\lambda$를 주는 방식을 사용하였습니다. 레이어가 1층부터 10층까지 있다고 했을 때, 더 높은 층으로 갈수록 더 적은 학습률을 할당하였습니다.

$$\theta_{t}^{l} = \theta_{t-1}^{l} - \lambda * J_{\theta_{l}^{'}}(\theta)$$

$$n^{l-1} = n^{l} / 2.6$$

#### Slanted triangular learning rates
딥러닝 모델을 학습시킬 때, learning rate을 똑같이 주거나, 점점 decay 하는 방식을 많이 사용합니다. 하지만, 해당 논문에서 저자는 기울어진 삼각형의 모양을 띈 learning rate을 LM 모델에 적용하였습니다. 공식은 아래와 같습니다.

$$cut = [T * \text{cut frac}]$$

$$ p = t/cut,  if\ t < cut$$

$$p = 1 - \frac{t - cut}{cut * (1/cut frac -1)} ,otherwise $$

$$\lambda_{t} = \lambda_{max} = \frac{1+p*(ratio - 1)}{ratio}$$


공식이 이것저것 나왔지만, 원리는 간단합니다. 우선 $T$는 training iteration의 횟수입니다. epoch 를 의미하는 것이 됩니다. $cut frac$은 전체 $T$에서 얼만큼의 iteration fraction이 진행되었을 때, learning rate가 치솟을지에 대한 정보이고, $cut$은 언제 learning rate이 감소하게 될지 그 iteration을 의미합니다. $p$는 얼마나 learning rate이 증가 및 감소할 지에 대한 ratio이고, 마지막으로 $ratio$는 max learning ratio 대비 min learning ratio 를 의미합니다.

###  Target task classifier fine-tuning

text classication의 경우 많은 단어들이 들어있는 document를 읽어드리게 됩니다. 이 경우 LM의 마지막 출력값이자 fine-tuning layer의 입력값을 last hidden state로 할 경우 많은 정보 손실이 발생하게 됩니다. 이와 같은 문제를 해결하기 위해서, 저자는 마지막 hidden state와 전체 hidden state의 max 값, mean 값을 사용하였습니다.

$$h_{c} = [h_{T}:maxpool(H):meanpool(H)]$$

### BPTT for Text Classication  (BPT3C)
document classification에서는 긴 문서를 분류하게 되는데, 이 경우 BPTT의 특성에 따라서, gradient vanish 또는 explode 현상이 발생하게 될 수 있습니다. 이에 따라 저자는 BPTT3C라는 방법론을 제시하여, $L$ 길이의 문서를 $b$ 라는 fixed length로 나누고, 배치를 돌릴 때, gradient update를 tracking 하여 학습시키도록 하였습니다. (model is initialized with the final state of the previous batch;)

### Bidirectional language model
마지막으로 ELMO와 같이, uni-directional 이 아닌, bi-directional 모델을 사용하였습니다.
