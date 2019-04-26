---
title: "Attention Is All You Need"
date: 2019-04-29 08:26:28 -0400
categories: [NLP]
tags: [NLP,HS]
---

Transformer에 대한 논문, Attention is all you need [논문](https://arxiv.org/abs/1706.03762)에 대해 이야기해보겠습니다.

구현 코드는 최대한 빠른 시일 내에 업데이트하겠습니다!

## Abstract

지배적인 시퀀스 모델들(RNN,CNN)은 인코더와 디코더를 기반으로 한다. 이런 네트워크를 기반으로 하는 모델들은 어탠션 메카니즘을 함께 사용한다. 우리는 오직 어탠션 메카니즘만을 사용하는 새로운 아키텍쳐 Transformer를 제안한다.

## Introduction

RNN, LSTM, GRU 와 같은 네트워크들이 시퀀스 모델링(language modeling, machine translation)에 굳건하게 사용되고 있다. 이러한 모델들은 인코더 디코더의 아키텍쳐 내에서 이뤄지고 있다. Recurrent model들은 전형적으로 스텝마다 hidden state를 형성하는데, 이전 스텝의 히든 스테이트 $h_{t-1}$ 에 대한 함수를 형성해 현재 스텝의 $h_{t}$를 형성하는 형식이다. 이와 같은 연산은 내부적으로 시퀀스 연산이며, GPU의 병렬 처리가 적용되지 않는다. 따라서, 최근 연구에서는 factorization tricks와 conditional computation과 같은 연구들이 이러한 문제를 해결해오고 있으며, 후자의 경우 성능을 향상시켰다. 하지만, 시퀀스 연산에 대한 제약은 여전히 남아있다.

## Background

시퀀스 연산을 줄이겠다는 목표 아래 CNN을 기본 빌딩으로 하여 히든 표상을 병렬처리가 가능하게 하는 Extended GPU, ByteNet, ConvS2S 들이 생겼다. 이러한 모델에서 두 임의의 입력 또는 출력 위치에서 신호를 연관시키는 데 필요한 작업 수는 ConvS2S의 경우 선형 적으로, ByteNet의 경우 대수적으로 위치 간 거리가 증가한다. Transformer의 경우 이러한 연산을 상수로 줄였다.

Self-Attention, intra-Attention이라 불리는 어탠션 메카니즘은 시퀀스의 표상을 학습시키기 위해서, 하나의 문장에서 다른 위치에 놓여있는 단어들 간의 어탠션을 적용하게 된다.

## Model-Architecture

대부분의 훌륭한 인공신경망 시퀀스 변환 모델들은 인코더-디코더 구조를 따른다. 인코더 디코더에서 인코더는 표상된 인풋 데이터 $[x_1,...,x_n]$을 연속적인 시퀀스 표상으로 $[z_1,..,z_n]$ 매핑시킨다. $z$를 가지고 아웃풋 시퀀스 $[y_1,..,y_n]$ 을 한 번에 하나씩 만들어 낸다.

Transformer 도 이러한 아키텍쳐를 따르는다. stacked self-attention과 point-wise, fully connected layer를 통해 인코더 디코더를 형성한다.

<img src = "/images/post_img/markdown-img-paste-20190426171325624.png">

**Encoder and Decoder Stacks**

- 인코더 :  동일한 레이어들을 6개 쌓은 형태로 이뤄져 있다. 각각의 레이어로 구성되어 있다. 첫 번째는 multi-header self-attention mechanism이고 두 번째는 간단한 position-wise fully connected feed-forward network이다. 또한 우리는 residual connection과 layer normalization을 각각의 하위 레이어에 적용한다. $\text{LayerNorm}(x + \text{Sublayer}(x))$ $\text{Sublayer}$란 함수에 적용되는 input 자체를 의미한다. residual-connection을 적용하기 위해, 모델의 모든 서브 레이어들의 히든 레이어 차원은 임베딩 차원 $d_{model}$ 512 와 동일하다.


- 디코더 : 인코더와 같이 동일한 레이어들이 6개 쌓은 행태로 이뤄져 있다. 각각의 레이어가 2개의 서브 레이어로 이뤄져 있던 인코더와는 달리 Masked Multi-Head Attention이라는 세 번째 서브 레이어가 존재한다. 인코더와 같이 residual connection을 적용한다. 또한 디코더의 어탠션 메커니즘에서 이후에 나오는 단어들과의 어탠션 연산을 막기 위해서 마스킹을 적용한다. 즉, $i$ 번째 위치해 있는 단어의 예측을 위해서는 $i$ 보다 이전에 나온 단어들을 통해서 예측하기 위함을 보장하는 것이다. 즉, 기존의 시퀀스 모델의 auto-regression의 형태를 따르게 한다.

**Attention**

어탠션 연산의 실행은 쿼리(query), 키(key), 밸류(value)의 조합으로 이뤄져 있다. 결과값은 쿼리와 키 사이의 어탠션 연산에 따라 나오는 가중치(compatibility : 적합성)들의 가중합과 같다.

- Scaled Dot-Product Attention

<img src = "/images/post_img/markdown-img-paste-20190426172739337.png">

$$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

원래 어탠션을 계산하는 방법은 additive attention, dot-product attention이 있다.

$$f_{att}(\mathbf{h}_i, \mathbf{s}_j) = \mathbf{v}_a{}^\top \text{tanh}(\mathbf{W}_a[\mathbf{h}_i; \mathbf{s}_j])$$
$$f_{att}(h_i, s_j) = h_i^\top \mathbf{W}_a s_j$$
우리는 Dot-product attention을 사용하게 된다. 연산의 복잡도는 additive와 유사하지만, 많은 프레임워크들이 해당 연산을 지원하고, 병렬화를 통해 고속화가 가능하다.

$Q,K,V$는 각각 query, key, values 를 의미하며, $d_k$ 는 key의 차원을 의미한다. 어탠션 메커니즘은 i번째 단어에 대해서 i번째 단어와 j번째 단어의 적합성(유사도)를 계산하고 싶을 때, $Q_{i}$와 $K_{j}$의 코사인 유사도를 계산하게 된다. 즉, 쿼리가 중심이고 키가 비교하려는 대상이 된다. 이렇게 코사인 유사도를 계산한 이후에, 키값의 차원의 루트값으로 나눠주게 된다. 예로 들어서 64면 8로 나눠주게 되는 것이다. 이유는 dot product의 값이 너무 커졌을 때, softmax 의 그레이언트값이 너무 작아지는 것을 막기 위해서이다.

**Multi-Head Attention**

<img src = "images/post_img/markdown-img-paste-20190426210945244.png">

하나의 어탠션 함수를 적용하는 대신, 쿼리(q) 키(k), 밸류(k)를 $h$ 번을 각각 다른, 학습된 선형공간 $d_{q}, d_{k}, d_{v}$ 에 투영하는 것이 학습에 효과가 있다는 것을 발견하였다. h개 만큼의 다른 공간에 투영된 쿼리, 키, 밸류들은 병렬적으로 어탠션 메커니즘이 적용되고 $d_v$ 의 차원을 가지는 아웃풋 값들을 반환한다. 즉, $d_v$ 가 $h$개 만큼 생성이 되면, 이를 concatenate해주고, 다시 한번, projection해주면서 결과값을 반환한다. multi-head attention  모델이 상이한 위치의 상이한 표현 부분 공간으로부터의 정보에 공동으로 참석할 수있게한다. 단일주의 머리를 사용하면 평균화가 이를 억제한다. 즉, 다양한 subspace에서 어탠션을 진행함으로써, 다양한 정보들을 수렴할 수 있게 된다.

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_{1},..,head_{h})W^O$$
$$\text{where } head_{i} = \text{Attention}(QW_{i}^Q,KW_{i}^K,VW_{i}^V)$$

해당 연구에서는 우리는 $h$ = 8 의 멀티 헤드 어탠션 연산을 사용했다. (embedding dimension이 512였으니, 각각 64 차원에 대해 어탠션을 적용한 것입니다.) 각각의 헤드에 대해, 차원이 줄었기 때문에, 헤드가 1이였을 때와 총 연산은 유사하게 된다.

**Applications of Attention in our Model**

모델 Transformer는 멀티 헤드 어탠션을 3 가지 다른 방식으로 사용한다.
첫 번째는 인코더-디코더 어탠션 레이어이다. 쿼리는 디코더의 이전 스테이트에서 오고, 키와 밸류는 인코더에서 오게 된다. 해당 레이어는 디코더의 모든 위치가 인풋 문장의 전체 위치에 대한 어탠션 메카니즘을 적용하는 것을 가능케한다.
두 번째는 인코더의 self-attention 레이어이다. 쿼리, 키, 밸류가 모두 같은 문장에서 온다.
마지막으로는 디코더의 self-attention 레이어이다. 여기서 특징은, 오른쪽의 정보는 마스킹한다는 것이다. 이는 위에서 언급했듯, rnn 기반의 네트워크가 가지고 있는 auto-regression 특성을 보존하기 위함이다.

**Position-wise Feed-Forward Networks**

어탠션 서브 레이어에 추가해서, 인코더와 디코더의 레이어 각각은 각 포지션에 독립적으로 적용되는 fully connected feed-forward network가 있다. 이는 ReLU 활성화 함수와 두 개의 선형 변환으로 구성되어 있다.
$$\text{FFN}(x) = \text{max}(0,xW_{1} + b_{1})W_{2} + b_{2}$$

위의 식을 표현하는 다른 방법은 kernel size가 1인 두 개의 convolution 연산이다. input과 output의 차원이 512($d_{model} = 512$)이고, 내부 레이어의 차원($d_{ff}$)은 2048이다.

**Embeddings and Softmax**

다른 시퀀스 모델과 같이, 우리는 인풋 토큰과 아웃풋 토큰을 특정 $d_{model}$로 임ㅂ-데이한다. 또한, 디코더에서 다음 스텝의 토큰을 예측할 때, 이전 스텝의 선형 변환(linear transform)과 소프트맥스(softmax)값을 사용한다. 두 개의 임베딩 레이어와 pre-softmax,linear transformation은 같은 가중치를 공유한다. 임베딩 레이어에서 가중치에 $\sqrt{d_{model}}$을 곱해준다.

**Positional Encoding**

우리의 모델은 recurrence와 convolution을 포함하고 있짐 않기 때문에, 토큰의 정보에 상대적 또는 절대적인 위치를 주입해야 한다. 따라서 우리는 "positional encodings"를 인코더와 디코더의 인풋 임베딩와 더해줌으로써 적용하였다.

$$PE_{(pos,2i)} = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$
$$PE_{(pos,2i+1)} = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$

## Why Self-Attention

<img src = '/images/post_img/markdown-img-paste-20190426204445819.png'>

위의 표에서 첫 번째는 레이어 당 총 연산 복잡도이다. 두 번째는 병렬처리가 가능했을 때 시퀀스 문제에 대한 요구 연산량이다. 세 번째는 네트워크의 장거리 종속성(long-range dependencies) 간의 경로 길이이다. 장거리 의존성을 학습하는 것은 많은 시퀀스 변환 작업에서 핵심 과제이다. 이러한 종속성을 학습하는 기능에 영향을 미치는 한 가지 핵심 요소는 경로 전 / 후 신호가 네트워크에서 통과해야하는 길이이다. 입력 및 출력 시퀀스에서 위치의 모든 조합 사이의 경로가 짧을수록 장거리 종속성을 쉽게 학습 할 수 있다. 따라서 서로 다른 계층 유형으로 구성된 네트워크에서 입력 위치와 출력 위치 사이의 최대 경로 길이를 비교한다. 결과는 위의 표와 같다.

추가적으로, self-attention은 해석 가능한 모델이다. 우리는 모델에서 주의 분포를 형성할 수 있고, 이에 대해 논의할 수 있게 된다.

## Training and Experiments Results

**Traning data and Batching**
<img src = "/images/post_img/markdown-img-paste-20190426205722151.png">
