---
title: "AWS 리눅스 서버 세팅 일기 (1)"
date: 2020-04-05 08:26:28 -0400
categories: [Engineering]
tags: [Engineering,HS]
---

[이전 포스트](https://hskimim.github.io/EBS_EFS_S3/)에서 야심차게, [Toy-project](https://github.com/hskimim/korean-stock-dashboard) 를 하는 팀원들이
사용하는 로컬 컴퓨터끼리를 연결해주는 NAS 를 AWS 서비스를 통해 구현해보겠다는 계획이 있었으나, 회사의 동료의 조언에 따라, 
EC2 instance 를 하나 만들고, EBS를 장착하기로 하였다. 그에 따른 과정을 정리해보려 한다.

___________

1. [EC2 instance 선택](##EC2 instance 선택)
2. Linux 사용자 계정 생성(##Linux 사용자 계정 생성)
3. 환경 설정(##환경 설)

___________

##EC2 instance 선택

어떠한 EC2 instance 를 써야, CPU 와 Memory 측면에서 불편함 없이 사용할 수 있으며, 비용을 최소화할 수 있을지에 
대한 고민이다. 요구 조건은 간략하게 아래와 같았다. 

- 3명 정도가 작업할 서버 (작업 시간은 보수적으로 같다고 설정)
- 데이터 수집을 하게 된 후, 분석을 어느정도 할 수 있는 서버
- 데이터의 양이 그리 크지는 않지만, 팀원들의 분석 과정이 memory-inefficient 할 것으로 간주

이에 따라, 후보군은 3가지 정도가 나왔다. 후보군은 [링크](https://aws.amazon.com/ko/ec2/instance-types/?nc1=h_ls)
를 참고하였다.



- r5.xlarge
    - vCPU : 4 
    - Memory : 32
    
- t3.xlarge 
    - vCPU : 4
    - Memory : 16

- t3.2xlarge
    - vCPU : 8
    - Memory : 32 
    
후에 [AWS Pricing Calculator](https://calculator.aws/#/createCalculator)를 사용해서, 1달 요금을 보니, 

- r5.xlarge 
    - 119.07 USD
    - 작성 시점 147,266.37 원
- t3.xlarge 
    - 79.14 USD
    - 작성 시점 97,880.75 원
- t3.2xlarge 
    - 155.28 USD 
    - 작성 시점 192,051.08 원
    
비싸다.. 그래도 이 계산기는 하루에 24시간 stop 없이 full 로 돌리는 것을 가정하니, 팀원들이 모두 대학생인 것을 감안하면, 
14시부터 02시로 설정해서 완벽하게 인스턴스 스케줄링이 된다는 가정 하에, 위의 결과에서 절반정도가 들게 된다. 팀원이 3명이니까,

- r5.xlarge 
    - 24,544 원
- t3.xlarge 
    - 16,313 원
- t3.2xlare
    - 32,000 원
    
그래도 대학생들에게 비싸긴 하다... 사실 여기서 그만 두고 싶었다.. 하지만 한번 끝까지 가보려 한다.
    
##Linux 사용자 계정 생성

좋은 [참고 자료](https://uroa.tistory.com/100)가 있어, 아주 쉽게 해결되었다. 

나, 팀원1/2 총 3명의 계정을 등록하고, home 폴더까지 셋팅이 끝났다.

##환경 설정

데이터 분석을 위한 이런 저런 설정이 많지만, 가장 기본적으로 pip virtualenv 를 사용하였다. 설치한 패키지는 
현재 내가 사용하고 있는, virtualenv 의 패키지를 그대로 넣어두었다.


