---
title: "Convolutional Neural Networks for Sentence Classification"
date: 2019-04-04 08:26:28 -0400
categories: [NLP]
tags: [NLP,HS]
---

김윤 교수님이 쓰신 Convolutional Neural Networks for Sentence Classification [논문](https://www.aclweb.org/anthology/D14-1181.pdf)에 대해 다뤄보도록 하겠습니다. 모든 내용과 이미지는 해당 논문을 참고합니다.

구현 코드는 [깃헙](https://github.com/hskimim/Natural_language_Processing_self_study/blob/master/CNN/) 사이트를 참고해주시면 감사하겠습니다

## Abstract

분류 문제를 해결하기 위해서 CNN(Convolutional Neural Networks)을 활용한 모델을 미리 학습된 단어 벡터들과 함께 실험해본다. 적은 수의 하이퍼파라미터 튜닝과 간단한 CNN 모델, 정적 벡터(static vectors : 자세한 설명은 뒤에서 계속 됩니다.)만으로도 훌륭한 점수를 벤치마크에 비해 내고 있음을 확인하였다.


## Introduction

최근 딥러닝 기술은 이미지 비전, 음성 인식에 있어서 큰 발전을 이룩해왔다. 이에 발맞춰, 자연어 처리 분야에서도 신경망 모델에 따른 단어 표현(word representation)기술의 발전에 따라 큰 발전이 있었다. One-hot-Encoding 이 가지는 sparse 문제와 high-dimensional 문제를 임베딩 기법을 통해서, 차원을 줄여 dense 한 단어 표현식을 형성하였다. 이러한 dense 한 표상(representation) 덕분에, 단어들 간의 의미론적(semantic)인 특성을 임베딩시킬 수 있게 되었다.

우리의 연구는 (LeCun et al., 1998)의 CNN모델을 사용하였고, 처음에는 미리 학습된(pre-trained) 임베딩 벡터를 사용하여, 모델 학습 시에는 CNN 파라미터만 학습시켰다. 더 적은 하이터파라미터만 학습시켰음에도 불구하고, 간단한 모델은 뛰어난 성능을 보여주었다. 이에 따라, 우리의 연구는 모델 학습 시, 미리 학습된 임베딩 벡터를 사용하기를 권장한다. 특별 도메인에 따라 학습된 단어 표상은 이어지는 연구에서 계속 제시된다.

## Model

<img src = "/images/post_img/markdown-img-paste-2019040621305219.png">

위의 이미지에 따르는 학습 프로세스를 따랐다. 구체적으로 설명을 들어가게 되면, k-dimensional 단어 벡터를 형성하였다.(k는 임베딩 벡터의 크기를 의미합니다.) 또한, 모든 문장의 길이 패딩(padding)을 통해서 n으로 통일시켰다. 이에 따라서, 아래와 같은 식으로 하나의 문장들을 행렬의 형태로 표현할 수 있다. $x_{1:n}$의 차원은 [max_length, embedding_dim], `+`는 concatenate의 의미이다. 위의 그림에서는 가장 왼쪽의 행렬과 같다.

$$x_{1:n} = x_{1} + x_{2} + ... + x_{n-1} + x_{n}$$

위의 그림에서 왼쪽에서 두 번째를 보자. convolution filter가 합성곱을 한 후에, 벡터가 된 모습이 나타난다. 보다 구체적으로 이야기해보면 다음과 같다.

$$c_{i} = f(w * x_{i:i+h-1} + b)$$

위의 식에서 $f$는 hyperbolic tangent와 같은 비선형 방정식을 의미한다.

$$ c = [c_{1},c_{2},...,c_{n-h},c_{n+h-1}]$$

(n+h-1 이 되기 위해서는 stride가 max_length 인 n 보다 1이 작은 크기만큼으로 할당되어야 합니다.)
위의 식을 통해 나오는 c는 위의 이미지에서 왼쪽 두 번째에 그려져 있는 벡터 하나를 의미한다. 즉 $c_{n}$은 하나의 합성곱 연산을 통해 반환된 value를 의미한다.

$$c_{hat} = max(c)$$

위의 식이 나타내는 것은 max pooling 인데, 하나의 커널에 따라서, 하나의 벡터가 나오게 되는데, 해당 벡터에서 가장 큰 값을 가지는 인덱스 요소만 선택하는 것이다. 즉, 가장 중요한 특징을 포착하는 것을 의미한다. 따라서 차원은 벡터에서 스칼라형태가 된다.
(원문에서는 `process by which one feature is extracted from one filter`라고 되어 있는데 보다 직관적 표현이 될 것 같습니다.)

**1. Regularization**

정규화 과정을 위해서 우리는 마지막 바로 전 레이어(penultimate layer : 즉 fully connected layer전을 의미합니다.)에서 Dropout 과 함께 가중치 벡터에 $l_{2}$norm 을 적용하였다.

Dropout 이란, back-propagation 과정에서 , 특정 벡터의 요소를 사라지게 하는 것(masking)을 의미하는데, 이는 베르누이 분포(Bernoulli distribution)을 따른다. 이를 식으로 표현하게 되면, 아래와 같다.

$$y = w * z + b$$

z가 우리가 가지고 있는, max pooling 후의 concatenate한 벡터 요소이다. w 와 b는 linear function의 parameter이다.

$$y = w * (z * r) + b$$

위의 식에서 보면, $z$에 $r$이라는 새로운 term이 추가되었는데, 이 요소가 이항분포의 확률값이 되고, 이에 따라서, 벡터의 요소가 확률 변수에 따라, 마스킹되는 것이다.

가중치 벡터 $w$ 에 $l_{2}$ norm 을 적용하므로써 다시 스케일링을 해주는 것은, $$w_{norm_2} > s$$의 경우, w의 norm_2 즉, euclidean distance 를 $s$ 로 재조정해줌으로써, overfitting을 막는 것이다.

## Datasets and Experiemntal Setup

사용한 데이터는 아래와 같다.

- MR : Movie Reviews
- SST-1,SST-2 : Stanford Sentiment Treebank(movie review의 확장)
- Subj : Subjectivity dataset
- TREC : TREC question Datasets
- CR : Customer reviews

**1. Hyperparameters and Training**

- Optimizer : Relu ( rectified linear units)
- Loss function : Stochastic gradient descent
- Windows of Filter : [3,4,5]
- Number of Filter : 100
- Dropout : 0.5
- $l_{2}$ constraint : 3
- Batch size : 50 with shuffled
- test dataset is randomly select 10% of training data

## Model VAriations
(사실 해당 논문에서 가장 중요한 부분이 아닐까 싶습니다.)

- CNN-rand : word vector 가 랜덤 초기화되어, 학습을 통해 수정된다.
- CNN-static : word2vec 으로 미리 학습된 word vector이다. 학습을 통해 수정되지 않는다.
- CNN-non-static : 각각의 문제에 맞춰서 보다 세세하게 튜닝된 미리 학습된 word vector 이다.
- CNN-multichannel : 두 세트의 단어 벡터가있는 모델이다. 각 벡터 집합은 '채널'로 취급되며 각 필터는 두 채널에 적용되지만 그라디언트는 채널 중 하나를 통해서만 전달된다. 따라서 모델은 다른 정적을 유지하면서 한 세트의 벡터를 미세 조정할 수 있다. 두 채널은 word2vec로 초기화된다.

<img src = "/images/post_img/markdown-img-paste-2019040703043771.png">

## Results and Discussion

CNN-rand 를 통해, 랜덤으로 초기화된 word vector를 쓴 CNN모델의 성능은 잘 작동하지 않았다. 이에 반해, 미리 학습된 (CNN-static)모델의 성능은 매우 잘 작동하였다. 이러한 결과를 통해, 미리 학습된 단어 벡터의 사용은 성능에 도움이 되고, 좋다고 할 수 있다. 각각의 테스크에 미세 조정된 단어 벡터는 이에 더 나아간 성능을 보여준다는 것 또한 알 수 있다.(CNN-non-static)

**1. Multichannel vs Single channel models**

multichannel을 통해서, overfitting을 방지하는 효과를 기대하였지만, 눈에 띄는 효과는 보기 힘들었고, regularization의 절차가 요구된다.

**2. Static vs Non-static Representations**

초기화된 word vector보다 미리 학습된 word vector가 더 성능이 좋고, 이보다, 미세 조정된 word vector의 성능이 더 좋다.

$$CNN_{rand} < CNN_{static} < CNN-non_{static}$$

## Conclusion

별다른 하이퍼파리미터의 조정없이, 미리 학습된 word2vec을 사용하는 간단한 CNN만으로도 잘 작동한다.
