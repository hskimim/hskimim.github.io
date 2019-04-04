---
title: "First-class-function and High-order-function"
date: 2019-02-10 08:26:28 -0400
categories: [Python]
tags: [Python,HS]
---

본 포스팅은 해당 [블로그](http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%8D%BC%EC%8A%A4%ED%8A%B8%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%95%A8%EC%88%98-first-class-function/) 와 [스택오버플로우](https://stackoverflow.com/questions/27392402/what-is-first-class-function-in-python) 이외 다른 학습 자료를 기반으로 하였습니다.

```
본 포스팅은 단순히 개인 공부 정리 용도이며, 저자는 배경 지식이 많이 부족한 이유로, 최선을 다하겠지만 분명히 완벽하지 않거나, 틀린 부분이 있을 수 있습니다. 계속해서 수정해나가도록 하겠습니다.
```


### First-class-function

위에 주석을 단 stackoverflow 의 설명을 해석하면, 아래와 같습니다.
```
first-class-function은 별다른 함수가 아니다. 파이썬의 모든 함수는 일곱 함수이다. 일반적으로 일급 함수로 불리기 위해서는, 정수(integar)나 문자열(string)과 같은 객체를 인자로 받아 이를 연산하거나 다룰 수 있어야 한다.(manipulate). 사용자는 일급 함수에, 하나의 변수뿐 만아니라, 다수의 변수와 함수 자체를 변수로 받을 수 있다.
```

### High-order-function

말 그대로 함수 안에 함수가 있는 형식입니다.
```python
def func1(msg) :
  def func2() :
    print(msg + '!')
  return func2
```
와 같이, func1을 선언하면, 해당 함수가 func2를 반환하게 되는 것입니다. 결과적으로는 func2의 값이 출력됩니다. 쉽게 말해 high-order function은 다른 함수를 반환하는 함수를 의미합니다.

개인적인 이해를 돕기 위해, 참고한 블로그에 나온 코드 예시와 유사하게, 같은 문제를 해결하되, 각각의 함수로 코드를 짜보았습니다.

only first-class-function
```python
def hello_blah(say_hello,who):
  print('{},{}!'.format(say_hello,who))
```

with high-order-function
```python
def hello(say_hello):
  def blah(who):
    print('{},{}!'.format(say_hello,who))
  return blah()
```

아직, 어떤 것이 다르고 어떤 이점이 있는 지 확실히 이해가 가지 않지만, 두 함수를 직접 사용해 보면, 차이가 발생합니다.
```python
hello_blah('hi','hs') # first-class-function
"hi,hs!"
```

```python
hello_who = hello('hello') # 바깥쪽 함수인 hello에 인자를 전달한 후, 변수 hello_who 에 객체를 전달한다.
print(hello_who)
<function blah at 0x1007dff50> # hello_who 라는 변수는 wrapper function인 blah 에 대한 함수를 가진 객체가 된다.
hello_who('hs')
"hello,hs!"
hello_who('sh')
"hello,sh!"
```

위의 예시와 같이, high-order-function을 사용하게 되면, wrapper function에 대한 유지 보수에 대한 부분이 효율적으로 이뤄질 수 있다는 장점이 있게 됩니다.

해당 포스팅에서는, functional programming 에 대해서 다루기 보단, high-order function에 따른 wrapper function의 효율성에 대해서만 이야기하려 합니다. 다음 포스팅, closure(클로져)와 decorater(데코레이터)에 따른, functional programming 에 대해 다뤄보도록 하겠습니다.
