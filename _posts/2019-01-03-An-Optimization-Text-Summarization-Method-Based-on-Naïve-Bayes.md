---
title: "An Optimization Text Summarization Method Based on Naïve Bayes and Topic Word for Single Syllable Language "
date: 2019-01-03 08:26:28 -0400
categories: [Machine_Learning]
tags: [Machine_Learning,HS]
---

## An Optimization Text Summarization Method Based on Naïve Bayes and Topic Word for Single Syllable Language
[논문 출처](http://www.m-hikari.com/ams/ams-2014/ams-1-4-2014/haAMS1-4-2014.pdf)

```
본 포스팅은 단순히 개인 공부 정리 용도이며, 저자는 배경 지식이 많이 부족한 이유로, 최선을 다하겠지만 분명히 완벽하지 않거나, 틀린 부분이 있을 수 있습니다. 계속해서 수정해나가도록 하겠습니다.
```

### ABSTRACT

나이브 베이즈 분류기를 통한 문서 요약기에 관한 논문이다. 논문의 키워드로는 단일 음절 언어(single syllable language , 문서 요약 , 차원 축소 , 나이브 베이즈 , 지도 학습 , 자연어 처리 , 토픽 언어가 있다.

저자가 베트남 출신임이고 이에 따라서 Abstract 에도 Vietnamese 에 대한 자연어 처리에 대한 내용이 언급된다. 해당 논문에서는 나이브 베이즈 알고리즘과 토픽 단어 세트를 통해서 문서 요약을 실시한다고 나와있다.

(In this paper, we propose a
text summarization method based on Naïve Bayes algorithm and topic words set.)

### INTRODUCTION

인터넷 발전에 따라 많은 정보량을 가지게 되었지만, 그에 따라 정보를 받아드리는 비용이 증가하게 되었고 문서 요약이 요구되었다. 문서 요약은 자연어 처리의 하위분야(subfield)로써, 기존의 문서가 들어가면 짧은 텍스트를 반환하는 과정이다.

50년동안 문서 요약 기술이 개발되면서, 크게 두 가지 방법론으로 나뉘게 된다. 1) 지도 학습(supervised learning) 2) 비지도 학습(un-supervised learning)이다. 비지도 학습 방법론은 데이터 셋을 빌드하고 모델을 트레이닝하는 등 많은 시간 소요를 아껴주었지만,높은 점수를 가진 문서와 문서의 특성(features)들을 선형 조합(linear combination)하는 방법론을 사용하기 때문에, 문서 요약의 질(quality)가 높지 않다.

나이브 베이즈 분류기를 사용한 1995년에 단어의 빈도(word frequency)와 문장의 길이(length of sentence)라는 특성을 가지고 처음 시도되었고, 1999년에는 tf-idf 방법론이 사용되었다. 이때 성능 개선을 위해서 vocabulary 와 WordNet이 사용되었다. (WordNet은 lexical database in English로, 어휘 사전을 의미한다.) 나이브 베이즈 분류기를 통한 분서 분류는 처음으로 성공적으로 이슈된 방법론으로 간주되고 있다.

해당 논문은 중국의 한자와 같이 하나의 단어가 의미를 가지고 있는 단일 음절 언어(single syllable language)에 대한 문서 요약을 시도하는 논문이다. 또한, 방법론으로는 위에서 언급한 나이브 베이즈 분류기를 선택한다.

또한 전체적인 프로세스에서 segmentation tool 절차를 제거함으로써, 컴퓨터 연산의 복잡성을 줄이는 절차를 선택하였다.

<img src="/images/post_img/markdown-img-paste-20190103175919606.png">

### RELATE WORKS

문서 요약의 첫 번째 절차는 주요한 문장을 특성을 기반으로 인식하고 추출하는 과정이다. 주요 사용되는 특성으로는 1. 문장의 길이 2. 문장의 위치 3. 주제 단어의 출현 4. 키워드의 빈도 등이 있다. 비지도 학습을 기반으로 하는 문서 요약 기법이 나오고 있지만, 지도 학습 기반의 기법이 여전히 훨씬 많다.(Naive
Bayes, artificial neural networks, decision trees, SVM models, hidden Markov models.)

Kupiec et al(1995)은 논문에서 나이브 베이즈 모델을
사용하면서 이를 추출될 것인지 말 것인지를 의미하는 두 문장들로(클래스) 나누었다. Kupiec가 사용한 특성들로는 주제 단어(heading word라고 쓰였는데, 정확한 의미는 확실하게 모르겠다..:<) , 문장의 길이와 단어의 빈도(1999년 논문에서는 tf-idf가 쓰였다.)가 있었다.

2000년도에는 뉴럴 네트워크 모형이 사용되었고, 1000개가 넘는 전문가가 만든 라벨이 사용되었다. Hidden Markov Model(HMM)은 2001년에 사용되었는데, 특성들은 서로 의존적(dependence)하다는 아이디어 아래에서 시행되었다. HMM 모델에 사용된 특성으로는 1. 문장의 위치 2. 문장에 사용된 단어의 갯수 3. 문서의 특성을 고려한 단어들이 문장에 있는지 이렇게 3가지가 고려되었다.


### FEATURE REDUCTION
**Text Representation**

<img src="/images/post_img/markdown-img-paste-20190103182242418.png">

w{i,j} 에서 i는 하나의 문서 내에 있는 i 번째 문장을 의미한다. j는 i번째 문장 내에 있는 j번째 단어를 가리킨다. 표현 방식은 frequency와 같은 방법으로 사용된다. (scikit-learn의 Count vectorizer와 같은 형태

하나의 문서에는 수많은 문장들이 있다. 심지어 이 포스팅에도 많은 문장들과 그에 따른 많은 단어들이 있다. 그에 따라 위의 이미지와 같은 행렬은 크기가 매우 클 것이다. 이에 따라, 컴퓨팅 연산에 대한 부담을 줄이기 위해서, 특성의 차원을 줄여주는 과정을 거쳐야 한다.

**Methodology of feature reduction.**
Feature Selection은 머신러닝과 그와 관련된 분야의 주요 이슈이다. 실제 데이터는 너무 과다하고(rebundant) 관련성이 없으며(irrelevant) 정확도나 트레이닝 속도에 부정적 영향을 끼칠 수도 있다. 이에 따라 적절하게 제거되는 과정을 거쳐야 한다.


### SINGLE SYLLABLE TEXT SUMMARIZATION

**Features Selection.**
해당 논문에서는 세 가지 특성을 제안한다.
```
1. Information Significant : 해당 문장이 가진 단어의 토픽성을 나타낸다.
문장이 특정한 단어를 가지고 있을 때,
그 단어를 가지고 있는 문장의 갯수를 세고 전체 문장으로 scaling.

2. Amount of information in a sentence :
일반적으로는, 문장의 길이는 주요 특성으로 놓지만,
해당 논문에서는 해당 문장에 존재하는 topic word의 갯수로 두었다.

3. Position of sentence : 첫 번쨰 paragraph 에 있는 문장이
영향력있는 문장인 것은 일반적이다. 이에 따라 1/i i = paragraph 의 순서

```

**Naïve Bayes classification .**
나이브 베이즈에 대한 자세한 내용은 [이전 포스팅](https://finlabgroup.github.io/NaiveBayesClassifier/)을 참조하길 바란다. 여기서 prior probability P(s) 는 C(s) / C(w) 로써, 전체 문장 갯수에서 특정 클래스에 속하는 문장의 수를 나눈 것을 의미한다.

**Using Naïve Bayes classification for single syllable text summarization .**

나이브 베이즈 분류를 두 단계로 사용하는데, 바로 Training 단계와 Summarization 단계이다. 우리는 사람이 정의한 요약본을 기반으로 한 라벨 데이터와 입력 변수 데이터를 트레이닝한다.

- Training Phase :
    - 데이터 세트(문서)를 빌드한다.
    - 명사 추출을 위한 pos tagging 을 실시한다.
    - 사람이 각각의 문서에서 가장 문요한 문장들을 추출한다.
    - 두 개의 클래스로 나눈다. 하나의 클래스는 선택받은 1번 라벨 문장들이고 다른 클래스는 선택받지 못한 0번 클래스이다.
    - 1번 클래스 문장들과 0번 클래스 문장들에 대한 토픽 단어 정도들을 계산한다.

<img src="/images/post_img/markdown-img-paste-20190103185035193.png">

논문에서 문장에 대한 topic word를 계산하는 방법이다.

- Extraction phase
    - 기존의 텍스트 T
    - T라는 문서를 S 라는 문장으로 세분화시킨다. S = {s_1,s_2,...}
    - 각각의 문장들이 가지고 있는 각각의 특성 F_j에 대한 확률을 계산한다. 그 후, 나이브 베이지안 분류기를 돌린다. 이 때, 두 개의 클래스가 존재한다.
    - 만약 extracted probability > not extracted probability 이면, 1번 클래스가 된다.

<img src="/images/post_img/markdown-img-paste-20190103190200470.png">


위의 이미지는 Extraction algorithm으로 `TSBN`이라고 불린다.
