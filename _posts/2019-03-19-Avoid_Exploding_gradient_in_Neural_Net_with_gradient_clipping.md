---
title: "Gradient Clipping against Gradient Exploding"
date: 2019-03-19 08:26:28 -0400
categories: [Deep_Learning]
tags: [Deep_Learning,HS]
---

본 포스팅은 해당 [블로그](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/) 를 참조하고 나름대로 번역하여 만들어졌습니다. 개인 학습 자료입니다.

최근, 개인 공부로 인해 RNN seq2seq 모델을 pytorch로 구현하던 도중,

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
```

위와 같은 코드 라인을 만나고, 과연 clip이 무엇일까라는 생각이 들었습니다. RNN(recurrent neural net)을 공부하다보면, 자연스레 long term dependency를 해결하기 위해 LSTM , GRU 를 공부하게 되는데, 이유는 vanishing gradient로 인해서, 멀리 떨어져있는 단어들 간의 관계성을 잃어간다는 것에 있었습니다.

 그 이유는, RNN의 특성 상, 계속 recurrent 해갔기 때문인데, 여기서 recurrent 는 이전의 state가 다음 번의 state를 결정하는 데에 기여하는 HMM과 같은 아키텍쳐를 따르게 되기 때문입니다. 따라서, activation function의 derivation이 1보다 작은 수가 많이 나오는 sigmoid나 그것보다는 향상되었지만 tanh function을 사용한 상태에서, sequence의 길이를 길게 만든다면 충분히 gradient가 vanishing 되고, 이에 따라 이전의 기억을 잃게 되기 때문이라는 맥락입니다.

여기서 간과해서는 안되는 사실 중 하나는 gradient의 overflow 문제인데, vanishing 되는 것과 같은 이유로 recurrent 의 특징 상, accumulated chain rule(back backpropagation의 연산)을 사용하게 되고, vanishing 을 방지하기 위해, LSTM 을 사용하게 된다면 gradient 가 overflow 되는 현상이 발생할 수 있게 됩니다.

서론이 길었는데, gradient 가 underflow 또는 overflow 되면, network가 불안정하다는 것을 의미하고, 이러한 경우 학습이 느려지거나 안될 수 있기 때문에, 이를 조정해주는 과정이 필요합니다.

gradient exploding 은 아래의 세 가지 이유로 발생할 수 있습니다.
- learning rate을 너무 크게 잡아서, 가중치의 업데이트가 너무 산발적일 때(large weight updates)
- 데이터의 전처리가 충분히 되지 않았을 때
- loss function을 선택이 부적합해서, loss의 크기가 너무 크게 잡힐 때(large error values)

위의 원인을 잘 조정해준다면, exploding gradient 문제를 어느정도 잡아줄 수 있지만, recurrent neural net 과 같은 모델의 경우, time steps 즉, sequence length 가 길게 되면, accumulated operation 이 많아지게 되고, 이에 따라 explding 의 가능성이 높아지게 됩니다. 대표적으로 LSTM 모델이 이러한 경우가 됩니다.

이러한 경우에는 back-propagation 연산을 하기 전에, loss(error)의 미분값(derivative)를 조정해주는 과정이 들어가게 되는데, 조정하는 방법에는 두 가지가 있습니다.

- Gradient Scaling : error derivative 를 rescaling 해주는 것입니다. 그레디언트 벡터를 1로 normalizing 하는 경우가 일반적입니다.
- Gradient Clipping : gradient value 를 특정 minimum value 로 줄이거나 특정 maximum value로 크게 하는 방법입니다.

만약 gradient value가 너무 커지게 되는 overflow 현상이 발생하게 된다고 가정을 해봅시다. 하나의 모델 안에는 수많은 파라미터들이 있고 이에 따라 각각의 gradient 가 존재하게 되겠지요. 이들을 모아 gradient vector 를 형성하고, 이러한 값들을 accumulated product가 되는데, batch size 마다 back propagation 을 통해서 파라미터를 back propagation 을 통해서 업데이트해주게 됩니다.

이런 업데이트를 해주기 전에, gradient vector가 `[0.7,1.1,3.4]` 와 같이 존재한다고 했을 때, 이를 normalizing하게 되면, 평균과 분산을 사용해 나눠주면, [-0.73, -0.45, 1.18] 와 같은 값이 나오게 됩니다. 3.4라는 큰 gradient 가 1.18로 줄어든 것을 확인할 수 있습니다.

추가적으로 clipping 을 사용해볼까요? clip을 1로 두고 `[0.7,1.1,3.4]` 을 놓게 되면 min(x,1) 이 적용됩니다. 손쉽게 list comprehension으로 구현하게 되면

```python
clip = 1
[np.min(i,clip) for i in [0.7,1.1,3.4]]
```
위와 같이 됩니다. normalizing 기법과는 다르게 확실히 강제적으로(forcing) 그래디언트를 삭감시켜주는 느낌이 듭니다.
