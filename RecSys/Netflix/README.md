
# Matrix Factorization Techniques For Recommender Systems

## Introduction

- user와 가장 적합한 item을 매치하는 것은 user 만족도와 관심을 향상시키는 중요한 요소
    - 추천시스템에서 매우 중요하게 다루어야함 (개인 맞춤형 추천)
- e-commerce 리더인 Amazon과 Netflix는 그들의 데이터를 활용하여 추천시스템을 제공하고 있다.

## Recommender System Strategies

- 2가지 strategies

### 1. **Content Filtering approach**

- user , product 에 대한 profile을 생성 ⇒ 속성을 특징화
- movie profile : 장르, 배우, 인기 등
- User profile : 인구 통계 정보, 설문문항 답변 등
- 쉽게 수집할 수 없는 **external information**이다!

### 2. **Collaborative Filtering approach**

- explicit profile 없이, 이전의 행동에 의존하는 방법
- 핵심 문장 : Collaborative filtering analyzes relationships between users and interdependencies among products to identify new user-item associations
- 장점
    - domain free
    - 이해하기 어렵거나 profile하기 어려운 data aspects를 다룸


- 일반적으로, *cold start* 문제 존재 : new products and user를 다루지 못함


- 두 가지 주요 분야
  #### 1. neighborhood methods
: item이나 user 간의 관계를 통해서 neighborhood 기준으로 선호도 추정
<img src = "https://user-images.githubusercontent.com/28617444/124582407-f01aad80-de8c-11eb-9802-ef096924e9a9.png" width="70%">

    - item-oriented approach
    - user-oriented approach

  #### 2. **latent factor models**
: rating patterns로 추론된 20~100개의 factor로 item과 user를 나타내는 것
  <img src = "https://user-images.githubusercontent.com/28617444/124582372-ebee9000-de8c-11eb-92a2-aa7383d193c6.png" width="70%">


## Matrix Factorization Methods

- latent factor model의 succesful realizations
  - **characterizes both items and users by vectors of factors inferred from item  rating patterns**


- MF는 additional information을 통합할 수 있다!
    - explicit feedback 은 평점이나, 좋아요 수 등으로 데이터가 매우 희소함 ( **Sparse** )
    - 반면, implicit feedback 은 검색 기록, 마우스 클릭 등 데이터가 많음 ( **Densely filled Matrix )**

### A Basic Matrix Factorization Model

- User와 Item을 f차원의 joint latent factor space로 매핑한다.
    - item에 대한 user의 관심

    ![Untitled 2](https://user-images.githubusercontent.com/28617444/124582378-ed1fbd00-de8c-11eb-8fd6-3d47bca15d79.png)


- 어떻게 매핑할 것인가
    - **Singular Value Decomposition (SVD)**
        - 정보 검색에서 latent semantic factor를 식별하는데 우수한 기법을 적용
        - factoring user-item rating matrix

- 그러나, user-item matrix 결측치가 많이 존재 ⇒ overfitting
    - 결측치 대체 : 왜곡 위험, expensive

- **observed ratings만 사용하고 regularized model을 사용하자!**

    ![Untitled 3](https://user-images.githubusercontent.com/28617444/124582381-ed1fbd00-de8c-11eb-9c2b-19e4b10b2f17.png)
    - 규제 값 람다는 cross-validation에 의해 결정됨

## Learning Algorithms

### Stochastic Gradient Descent

- gradient 반대 방향

    $$e_{ui} = r_{ui} - q_{i}^Tp_u$$

- learning rate로 업데이트

  ![Untitled 4](https://user-images.githubusercontent.com/28617444/124582384-edb85380-de8c-11eb-9f11-1d4dc3e08c0e.png)


### Alternating Least Squares

- $$q_i$$와 $$p_i$$ 는 알려지지 않았으므로 convex 가 아님
    - **이를 해결하기 위해 둘 중 하나를 고정하는 것을 반복**
- 장점
    1. can use parallelization
    2. centered on implicit data
        - sparse하지 않으면, SGD(단일 훈련)은 실용적이지 않음

## Adding Biases

- Biases or intercepts 존재
    1. user마다 점수 주는 경향이 다름
    2. 특정 item마다 다른 item보다 더 높은 경향
- $$q_i^{T}p_u$$ **는 이를 해결할 수 없음**
    - **So identify the portion of these values that individual user or item biases can explain**
    - 데이터의 True 상호작용만을 고려하겠다
        - $$b_{ui} = \mu + b_i + b_u$$
    - 확장
        - $$r_{ui} = \mu + b_i + b_u + q_i^{T}p_u$$

            ⇒ Gloval average, Item Bias, User Bias, User-Item interaction 4가지 고려 가능

    - minimize squared error function
    ![Untitled 5](https://user-images.githubusercontent.com/28617444/124582388-edb85380-de8c-11eb-8f83-6412ed7f4ca8.png)


# Additional Input Sources

- *Cold Start* problem
    - **incorporate additional sources of information about the users**
    - implicit feedback을 사용하자
1. $$N(u)$$의 item에 대한 user의 암시적 선호도

  ![Untitled 6](https://user-images.githubusercontent.com/28617444/124582392-ee50ea00-de8c-11eb-8d58-c1ed482e1137.png)

- 총합 normalize

    ![Untitled 7](https://user-images.githubusercontent.com/28617444/124582393-ee50ea00-de8c-11eb-919a-ca9969135e5c.png)

2. user에 대한 특성

  ![Untitled 8](https://user-images.githubusercontent.com/28617444/124582396-eee98080-de8c-11eb-89e5-e8fd7dcfabce.png)


- matrix factorization model (integrate all signal sources with enhanced user representation)

  ![Untitled 9](https://user-images.githubusercontent.com/28617444/124582397-eee98080-de8c-11eb-99cd-98997d57e008.png)

## Temporal Dynamics

- 시간적 측면을 반영

  ![Untitled 10](https://user-images.githubusercontent.com/28617444/124582400-ef821700-de8c-11eb-8360-26eba7d2c568.png)
    1. bias $$b_i$$
        - 배우의 신작 출현 ⇒ 시간이 지나면 흥행이 줄어듬
    2. parameter $$b_u$$
        - 예전에는 평균 4의 평점, 요즘 평균 3의 평점
    3. user factors (vector $$p_u$$)
        - 선호도가 변함 ( 스릴러 ⇒ 범죄)
    - 아이템의 경우, 시간이 지나도 static 하기 때문에 (t)가 반영되지 않음

## Inputs With Varying Confidence Levels

- **not all observed ratings deserve the same weights or confidence!**
    - 장기적 특성 반영하지 못하는 아이템 ⇒ 평점 영향
    - 한쪽으로만 기우는 적대적인 user
    - implicit feedback : 수량화하기 어렵기에 "probably" 의미의 Cruder Binary 표현됨


- 따라서, 빈도를 나타내는 수치의 신뢰도를 도입
    - matrix factorization model can readily accept varying confidence levels, which let it give less weight to less meaningful obeservations

    ![Untitled 11](https://user-images.githubusercontent.com/28617444/124582402-ef821700-de8c-11eb-9d9c-86793ea1fdff.png)


# Conclusion

- Matrix Factorization technique - dominant methodology within collaborative filtering recommenders
    - accuracy superior to classical nearest-neighbor techniques
    - compact memory-efficient
- 모델을 간편하게 해주는 방법론 제시
    1. Multiple forms of feedback
    2. Temporal dynamics
    3. Confidence levels

## References
[1] [Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009): 30-37.](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
