---
title: "Improving Language Understanding by Generative Pre-training"
date: 2019-08-26 08:26:28 -0400
categories: [NLP_paper]
tags: [NLP_paper,HS]
---

Transfer Learning 이 핫하다! 라고 말하기도 이미 너무 늦어버린 시기가 되었네요. Transformer 의 등장 이후, Attention의 시대가 시작되었고, 성능 향상을 위해 Semi-supervised learning이 대두되던 때에, Open-AI 에서 선보였던 역작, gpt1이라고도 불리는 [논문](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 에 대해 간략한 리뷰를 진행해보고자 합니다.

NLP를 급부상시킨 지금도 굳건한 영웅인 word2vec은 언어를 어떻게 표현(representation)해야 하는 지에 대해 훌륭한 방법을 제시한 방법론입니다. 이와 같이, 자연어처리에서 성과를 부스트시켜주는 요소 중 언어 표상(language representation)은 매우 중요한 요소로 인지되고 있습니다.

이 글을 읽으시는 많은 독자분들도 이미 아시다시피, 한 단어에 대해 하나의 벡터만을 지원하는 word2vec의 한계를 극복하기 위해, 보다 깔끔하게는, language representation을 word-level이 아닌 context-level로 가져오기 위해서, 다양한 방법들이 제시되었습니다. 대표적으로 [ELMO](https://hskimim.github.io/Deep-contextualized-word-representations/) 와 ULM-Fit 이 있습니다.

위의 방법론은 word-level information에서 무언가를 더 이끌어내기 위해서, un-supervised learning을 사용했다는 점에서 공통점이 존재하는데, 몇 가지 이슈들이 존재하였습니다.

1. 어떻게 사전 학습시키는 것이 최선인가?
  - unsupervised learning으로 학습시키는 데에 있어서, 어떤 최적화 목적 함수를 두어야 하는지에 대한 문제입니다.
  - machine translation, language modeling, discourse coherence 등 다양한 테스크에 대해 언어를 사전 학습시키는 연구들이 있었습니다.

2. 이렇게 학습시킨 언어를 어떻게 전달(transfer)하는 것이 최선인가?
  - 이 또한 다양한 방법이 있는데, 모델 아키텍처를 테스크에 맞게 변형하거나, learning objective 를 추가하거나, 변형하는 방법들을 사용하였습니다.

위와 같은 이슈들은 semi-supervised learning을 통한 NLP 모델의 효과적인 성능 개선에 뚜렷한 길을 제시하지 못하게끔 하였습니다.

그렇게 나온 것이 GPT1입니다.

![스크린샷 2019-08-27 오전 2.02.40](/images/post_img/스크린샷%202019-08-27%20오전%202.02.40.png)

위의 사진이 의미하는 것은 다음과 같습니다.

classification, similarity 등등의 다양한 테스크에 대해서 모두 적용할 수 있으며, 이는 비지도 학습 시, 입력 데이터에 약간의 변형을 주면 됩니다. 비지도 학습은 language modeling을 사용하고, 이는 transformer 의 decoder 부분을 사용합니다. 마지막으로 예측을 해야하는데, 이 부분은 마지막의 linear (projection layer)를 통해서 간단하게 구현이 가능합니다.

이는 위에서 언급한 두 가지 문제에 대한 대답이 될 수 있습니다. 첫 번째로 사전 학습에 적합한 목적 함수에 대한 이슈에 대해서는 transformer의 decoder 의 성능을 통해 robust하게 일원화시킬 수 있다는 것입니다. 두 번째로는 추가 또는 결합을 통해 사전 학습의 정보를 이전(transfer)해오던 것을 간단한 projection layer를 접목하므로써, 모델 아키텍처 자체의 변형을 최소하도록 한 것입니다.

이제부터 모델의 전체적인 구조에 대해서 알아보도록 하겠습니다.

## Framework

1. 비지도 학습

위에서 언급드렸던 것과 같이, 비지도 학습을 통한 문맥 정보는 language modeling을 통해 진행됩니다. 이는 transformer의 decoder 부분에 해당되는 부분입니다. 그 이유는 아시다시피, language modeling의 목적이 그 다음에 발생하는 단어를 맞추는 것으로, masking이 되어있어야 하기 때문입니다.

$$h_{0} = UW_{e} + W_{p}$$
$$h_{l} = \text{transformer block}(h_{i})  i \in [1,n]$$
$$P(u) = \text{softmax}(h_{n}W_{e}^{T})$$

위의 식을 보면, transformer 논문에서 사용하였던 방법론과 같음을 알 수 있습니다. positional encoding을 사용함으로써, 위치 정보를 넣어주었고, transformer decoder 부분을 사용해주었으며, 이렇게 나온 $h_{n}$ 을 projection layer $W_{e}$ 를 통해 값을 예측하게 됩니다. 여기서 예측하는 것은 마지막 예측값 (감성분석일 경우 pos or neg)이 아닌, 언어 모델의 그 다음 단어를 의미합니다.

각각의 notation을 하나씩 짚어보면, $W_{e}$은 embedding matrix를 의미하고, $W_{p}$ position embedding matrix입니다. $n$은 layer 의 갯수를 의미합니다. 요약해보자면, $n$개의 layer를 순차적으로 올라가면서, context information을 학습하며, $n=0$ 즉, 실제 단어가 입력될 때는 positional encoding이 된 상태에서 들어갑니다. 이렇게 최종 레이어까지 나온 단어는 문맥 정보를 반영하게 되고, 이를 embedding matrix와 곱해줌으로써, 하나의 단어에 대한 context vector가 됩니다.

2. 지도 학습

이 부분은 매우 간단합니다. 테스크에 따라 약간씩 변형된 형태의 projection layer를 attach해주면 됩니다.

$$P(y|x^{1},...,x^{m}) = \text{softmax}(h_{l}^{m}W_{y})$$

위의 식은, 어떤 인풋 시퀀스가 들어갔을 때, 정답 y가 나올 확률입니다. softmax를 통해 이러한 log probability를 최대화하는 방법으로 학습이 됩니다.


3. Model specifications

마지막으로 실제 저자가 적용한 모델의 하이퍼 파라미터에 대한 정보에 대해 알아보겠습니다.
사실 pytorch-transformer 가 나왔고, 8-GPU 로 1달이 걸렸다고 하니, 실제 적용은 힘들겠네요...ㅠㅠ

- transformer decoder num_layer : 12
- decoder hidden_dim : 768
- multi_head_num : 12
- position-wise feed-forward hidden_dim : 3072
- optimizer : Adam (with max lr : 2.5e-4)
- batch_size : 64
- epoch : 100
- max_len = 512
- weight_initialization : N(0,0.02)
- dropout_rate = 0.1
- L2 regularization : 0.01(lambda)
- activation function : GELU
- tokenizer : spacy

### Performance

![스크린샷 2019-08-28 오전 2.50.29](/images/post_img/스크린샷%202019-08-28%20오전%202.50.29.png)

위의 표는 classification 과 semantic similarity 에 대한 결과 표입니다. Transformre LM 에 간단한 fine-tuning 만으로, 이전의 앙상블 모델들 대비 좋은 성과를 보임을 알 수 있습니다.

읽어주셔서 감사합니다. 피드백은 언제나 환영입니다!! (more than welcome!)
