---
title: "Database vs File system storage"
date: 2020-03-23 08:26:28 -0400
categories: [Engineering]
tags: [Engineering,HS]
---

개인 프로젝트를 진행하면서 데이터를 이리저리 수집해가던 와중에, 조언을 구하는 지인으로부터 데이터들을 데이터베이스에 저장하여 관리하는 것이 어떻냐는 조언을 듣게 되었다.
부끄럽게도 이전까지 데이터베이스를 통해 데이터들을 저장하는 방법을 사용해온 적이 없던터라 관련 자료를 알아보던 도중, 문득 왜 file system 이 아니라 DB이어야 할까라는 고민이 들었다.
이에 따라, Database vs File system storage 라는 주제에 대해 알아보고 정리한다.

참고자료 : [stackoverflow](https://stackoverflow.com/questions/38120895/database-vs-file-system-storage), [blog](https://dzone.com/articles/which-is-better-saving-files-in-database-or-in-fil)

### 데이터 베이스의 장점

- [ACID](https://ko.wikipedia.org/wiki/ACID) 일관성 
    - ACID 는 데이터 베이스 내에서 파일의 I/O (transaction : 트랜잭션) 과정이 안전하게 수행되는 것을 보장하기 위해 필요한 성질을 의미한다.
        - **A**tomicity(원자성) : 트랜잭션과 관련된 일들이 실행되다가 중단되면 안된다. 즉, 하나의 트랜잭션은 성공, 실패를 함께 해야한다.
        - **C**onsistency(일관성) : 데이터베이스 내의 데이터에 대한 무결성 조건을 위한하는 트랜잭션은 중단되는 것을 의미한다. 즉, 조건에 반하는 트랜잭션은 실행될 수 없다.
        - **I**solation(고립성) : 트랜잭션 과정의 데이터를 확인하거나 다른 트랜잭션이 끼어들 수 없다. 고립성은 성능관련 이슈로 유연하게 적용된다고 한다.
        - **D**urabuility(지속성) : 적용된 트랜잭션은 영원히 반영되어야 한다. 예로 들어 커밋된 git이 version control로 인해 다시 이전으로 돌아가도 이전의 commit log는 지속적으로 남게 되는 것과 유사하다.
        
- 상대적으로 안전하게 데이터를 저장할 수 있다.
- ACID의 D로 인해서, 파일이 DB에 동기화되고 이에 대한 로그를 추적할 수 있어 관리가 용이하다.
- 백업이 지원된다. 
- 서버-클라이언트 관계에서 서버 쪽에서 데이터를 공급해주는 것이 용이하다.

### 데이터 베이스의 단점 

- 음성 파일, 이미지 파일 등을 [blob](https://en.wikipedia.org/wiki/Binary_large_object)의 형태로 저장해야 한다.
    - blob 이란 말 그대로, 바이너리 처리된 데이터이다. 파이썬의 pickle data format 과 유사한 느낌이다.
- 데이터베이스의 백업 작업은 무겁다. -> file system 또한 가볍지 않을 것 같다. 
- 메모리 비효율적이다.
    - RDBMS는 메모리를 사용해서 모든 데이터는 RAM에 얹어진다. 따라서 정렬, 인덱싱과 같은 쿼리문이 RAM에 실려서 많은 데이터를 다룰 경우 많은 리소스를 요구한다.
- ACID 를 통해 relational mapping을 사용해야 하는데, 모든 데이터에 대해 이러한 관계 설정이 어렵다.
 
### 데이터 베이스는 이럴 때 적절하다.

- 데이터가 구조화(structured)되어 있을 때
- 데이터 간 관련성(relateness) 가 존재할 때
- 데이터의 보안이 요구될 때
- 작업에 필요한 데이터 갯수가 많지 않을 때

### 파일 시스템의 장점 

- 데이터의 위치가 잘 구조화되어 있다면 성능이 DB보다 좋을 수 있다. ex) `Select *` 쿼리를 통한 데이터 서치는 매우 느리다.
- 수집한 데이터를 읽고 쓰는 것이 훨씬 간단하다.
- 데이터를 마이그레이션(migration) 하는 것이 더 쉽다. 
    - cloud storage 에 마이그래이션하는 것 또한 쉬워진다.  
- ACID 와 같은 무결정 보장 operation이 없기 때문에, 데이터를 트랜잭션하는 데에 별 신경을 쓰지 않아도 된다.

### 파일 시스템의 단점 

- DB이 경우 ACID operation을 통해 데이터의 무결성을 보장해주지만, file system의 경우 특정 데이터가 없거나 해킹되었을 때 이를 보장해주는 장치가 없다.
- 상대적으로 덜 안전하다.

### 파일 시스템은 이럴 때 적절하다. 

- 큰 데이터가 요구될 때
- 파일의 트랜잭션에 많은 사용자가 존재할 때  


### 정리

아직도 크기가 큰 데이터에 대해 DB보다 file system이 더 적절한지 이해가 충분히 가지 않는다. 
이 과정에서 DB의 transaction overhead, data duplication 등 많은 용어들이 나오는데, 이는 DB 자체에 대한 공부가 선행되어야 할 것 같다. 