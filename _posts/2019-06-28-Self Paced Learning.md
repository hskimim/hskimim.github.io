---
title: "Self-Paced Learning for Latent Variable Models"
date: 2019-06-28 08:26:28 -0400
categories: [Deep_Learning]
tags: [Deep_Learning,HS]
---

Curriculum Learning 의 시발점이 된 Y.Bengio 교수님의 논문에 대해 직전 포스팅에서 다뤄보았는데요. 이번에는 쉬운 데이터부터 점차 데이터의 난이도를 고도화하자는 아이디어를 공유한 상태에서 방법론의 변화를 준, Kumar 교수의 2010년 논문, ["Self-Paced Learning for Latent Variable Models"](http://ai.stanford.edu/~pawan/publications/KPK-NIPS2010.pdf) 에 대해 이야기해보려 합니다.

_________________

이전 논문인 Curriculum Learning 에서는 실험자가 직접 쉬운 데이터셋을 명시하고 순서대로 모델에 학습시켰습니다. 하지만 실험자가 말하는 쉬운 데이셋이라는 기준이 모호하고, 일종의 라벨을 달아야 하는 과정이기 때문에, 실질적으로 작동하기 어렵다는 한계가 있었죠. 이에 따라서, 모델이 학습함과 동시에 쉬운 데이터셋을 일정 임계치에 따라 선택하는 방법론인 Self-Paced Learning 이 출현하게 되었습니다.

해당 논문의 모델은 Support Vector Machine 을 사용하였고 모든 모델은 loss function 이 나오기 마련입니다.

$$w_{t+1} = argmin_{w \in \mathbb{R}^d}(r(w) + \sum_{i=1}^n \ f(x_{i},y_{i};w))$$

위의 식은 다들 아시다시피, regularization term $r(w)$ 이 포함된, loss function 입니다. Self-Paced Learning 은 이러한 loss function term 에 하나의 항을 덧 붙이는데 이는 아래와 같습니다.

$$w_{t+1} = argmin_{w \in \mathbb{R}^d , v \in \{0,1\}^n}(r(w) + \sum_{i=1}^n \ v_{i}f(x_{i},y_{i};w) - \frac{1}{K} \sum_{i=1}^n v_{i})$$

새로 추가된 항들을 하나씩 살펴보도록 하겠습니다. 첫 번째로, $v \in \{0,1\}^n$ 입니다. 쉽게 말해 v 는 이진수입니다. 직관적으로 생각해 유추해보면, 쉬운 데이터일 경우 1이라는 불이 켜지고 아닐 경우, 0으로 꺼지는 형식이 될 것 같네요. 두 번째로는 $v_{i}f(x_{i},y_{i};w)$ 입니다. 아까 말씀드린 바와 같이, 쉬운 데이터라고 한 부분만 loss function 이 계산되는 형식입니다. 모델이 감당할 수 있는 데이터 (수준에 맞는 데이터 즉, 쉬운 데이터) 일 경우, loss function 이 작게 계산될 것이기 때문에, 최적의 $v_{i}$  를 찾게 되겠습니다. 마지막으로 $- \frac{1}{K} \sum_{i=1}^n v_{i}$ 입니다. 앞에 음수가 취해져있는 것으로 보아서, 뒤의 부분이 커질 수록 loss 가 작아지게 됩니다. 중요한 것은 바로 분모에 위치해있는 $K$ 입니다. 실제 프로세스에서는 $K$ 의 값이 점점 커지게 구성이 되어 있습니다. 이에 따라 $\frac{1}{K}$ 이 0으로 가까워 짐으로써, 새로 생긴 term의 영향이 점점 작아지게 되는 것이죠. 이에 따라 $v_{i}$ 에 1이 자연스레 많아지게 됩니다. 즉, 전체 점차 모든 데이터를 학습하게 됩니다.

그렇다면 이제 어떻게 최적화를 시킬지에 대해서 이야기해볼까요? 논문에서는 alternative search strategy 라는 방법론을 통해 최적화를 시킨다고 합니다. 말이 어렵지 사실은 간단합니다. 바로 위에서 나온 loss function 에서 우리가 최적화해야 할 변수는 두 가지입니다. $(w,v)$ 하나를 최적화할 때, 다른 하나를 고정시킨 상태에서 최적화를 하는 것이 바로 alternative search strategy 이 아이디어입니다. $w$ 를 최적화해야 한다고 했을 때, 사용자는 $v$ 를 고정하게 되고, 이는 저희가 원래 알고 있던 loss function과 똑같게 됩니다. (데이터의 수만 fractional 합니다.) $v$ 를 고정한다고 했을 대, 사용자는 $w$ 를 고정하게 되고, 아래와 같은 식을 풀게 됩니다.

$$v_{i} = \pi(f(x_{i},y_{i};w) < \frac{1}{K}), \text{where} \ \pi(.)$$

위의 식에서 $\pi$ 는 indicator function이라는 것으로, 특정 데이터 $v_{i}$가 임계값을 만족하는지에 대한 함수라고 할 수 있습니다. 이전에 말씀드린 바와 같이 $K$ 는 monotonic 하게 증가하고 이는 아래의 식을 따릅니다.

$$K \rightarrow \frac{K}{\mu}$$

<img src = '/images/post_img/markdown-img-paste-20190626150901906.png'>

알고리즘은 위와 같습니다.

<img src = "/images/post_img/markdown-img-paste-2019062615094192.png">

SVM 의 최적화 방식인 CCCP (baseline) 과 비교한 성능 표로 Test Error 에서 Self-Paced Learning 방법론이 우위를 띄고 있음을 알 수 있습니다.

이전에 다뤘던 CL 방법론과 SPL 의 공통점은 데이터를 쉬운 것부터 순차적으로 학습시키자는 기본 아이디어에서 시작한다는 것이고, 차이점으로는 CL의 경우 실험자가 prior knowledge 를 적용한다는 점, SPL의 경우, model 의 최적화 과정에서 적합한 샘플을 고른다는 점에서 차이점이 있습니다.
