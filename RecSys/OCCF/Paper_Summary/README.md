# Collaborative Filtering for Implicit Feedback Datasets

## Abstract
---

- 추천시스템 : to improve customer experience based on **prior implicit feedback**
    - EX) 구매 이력, 검색 기록, 클릭 등
    - 직접 설문조사를 입력한다든지 등의 direct input이 아니기에 상품을 고객이 싫어할지 안할지 실질적인 증거를 찾기 힘듬
- 해당 논문 :  identify unique properties of implicit feedback datasets
    - confidence level를 도입함으로써 positive, negative preference를 다루고자 한다.
    - implicit feedback recommender에 적합한 factor model 제안
    - also suggest a scalable optimization procedure

## Introduction
---

- e-commerce 인기 높아짐 ⇒ 추천 시스템 구축하는 것이 중요해짐
    1. Content based approach
        - 각각의 user나 product에 대해 profile을 만든다.
        - ex) Movie profile - genre, actors, popularity
        - ex) User profile - 인구 통계 정보, 설문조사 문항
        - **그러나, 이러한 external information은 수집하기 매우 힘들다!**
    2. **Collaborative Filtering (CF) - 논문에서 사용한 방식**
        - explicit profile 생성 없이, past user behavior 에 의존한다.
        - 도메인을 타지 않는다는 장점 ⇒ content based approach 통해 profile 하기 어려운 데이터도 다룰 수 있다!
        - *cold start problem* 가 일반적으로 존재 : new product나 user를 다루지 못함

### 2 types of input in 추천시스템

- explicit feedback
    - ex) Netflix start 수, 좋아요 수 등
    - 그러나 이용하기 어렵다. (일반적으로 많이 안 하니)
- abundant implicit feedback을 통해 user preference를 추론하는 것이 중요함!
    - ex) 구매 이력, 검색 패턴, 클릭 등
    - 같은 저자의 책을 많이 구입했다면, 그 저자를 좋아할 것이기에
    - **따라서, implicit feedback을 처리하기에 알맞은 알고리즘을 실험하고자 함**

### Implicit feedback의 4가지 특징

1. No Negative feedback
    - User의 행동을 관찰하는 것은 그들이 좋아할 것 같다고 추론할 수 있다.
        - **그러나, 싫어할 것 같다고는 하지 못한다.**
        - "show를 보지 않음" 의 해석
            1. 실제로 싫어해서 안 본 경우
            2. 해당 show 존재 자체도 모른 경우
            3. 이용 불가능 (ex. 연령)
    - 그래도, explicit은 존재하는 값만 사용하는데 missing data도 다루고자 한다.
2. Implicit feedback은 inherently noisy
    - user behavior를 분석해보면, 그들의 preference를 **추측**할 뿐이다.
        - "구입을 한다" 의 해석

            : 선물로 구입한 것일수도 (그 사용자가 안 좋아하는 것이여도)

        - "특정 채널에서 계속 티비를 본다"의 해석

            : TV보다 잠든 것일수도

3. numerical value of implicit feedback ⇒ confidence 를 의미!
    - 즉, action의 빈도를 의미한다.
        - 얼마나 특정 show를 보았는지, 몇 번 샀는지
        - **값이 크다고, 높은 preference를 의미하는게 아님!**
            1. 가장 좋아하는 영화를 1번 봤을 수도 있으니
            2. 시리즈물이라서 계속 들어가서 봤을 수도 있고
        - 그래도, useful 하다고 할 수는 있다.
            - 특정 관측치가 있다는것은 신뢰도에 관해 얘기할 수 있으므로
            - 1번 본게 아닌, 반복해서 event가 발생한다면, more likely to reflect the user opnion.
    - 반면, numerical value of explicit feedback ⇒ preference 를 의미
        - 1 star : 별로 안 좋아함
        - 5 star : 매우 좋아함
4. implicit feedback 추천시스템의 적절한 평가지표가 필요
    - 평점 데이터는 예측값, 실제값 간의 차이를 평가하는 mean squared error
    - 그러나, 시청시간이나 클릭 수 등은 다음과 같은 상황을 고려해야 함
        1. 동시간에 방영되는 TV Show : 다른 프로그램 좋아한다해도, data가 쌓이지 않을 것
        2. 1번 이상 show를 보았을 때, 한 번 본 경우와 어떻게 다르게 평가할지

## Preliminaries
---

- $r_{ui}$
    1. Explicit feedback datasets :  user가 i 에 대한 선호도
    2. Implicit feedback datasets : user의 행동에 대한 observations
        - TV Show case1 ⇒ $r_{ui}$ = 0.7 : TV show의 70%를 시청했다.
        - TV Show case2 ⇒ $r_{ui}$ = 2 : TV show를 2번 시청했다.
        - missing one 을 무시할 수 없다 ( no action 이 0을 의미하므로!)

## Previous work
---

### 3.1 Neighborhood models

1. user-oriented
    - liked minded user의 rating을 통해 unknown rating을 추정하는 것
2. Item-oriented
    - more popular (더 의미 있음)
    - 유사한 item에 같은 사람의 rating을 통해 unknown rating을 추정하는게 상식적으로 더 의미 있을 것
    - 예측에 대한 원인을 설명 가능하다!
    - 가장 중요한 접근법은 **아이템 간의 유사도 측정, $s_{ij}$ (피어슨상관계수, 코사인 유사도 등등)**<br>
    <img src = "https://user-images.githubusercontent.com/28617444/125593156-538dfacd-0663-4812-91f6-df0d99646588.png" width="30%">
    - 이 방식은 explicit feedback에 실용적임
        - user , item에 따른 bias 추가 등
    - **그러나, implicit feedback datasets에 적용하기 매우 힘듬**
        - item의 빈도 : user마다 다 다른 scale ⇒ 유사도 구할 수 없음
        - user preference와 confidence를 구별하기가 쉽지 않기에

### 3.2 Latent factor models

- 관측된 rating을 설명하는 latent feature
    - SVD (user-item observations matrix)
        - user-factors vector
        - item-factors vector
        - 두 벡터의 내적 ⇒ $r_{ui}$
        - explicit feedback datasets에 적용하기 위해, 오직 관측된 ratings만 사용되어 왔다.
        <img src = "https://user-images.githubusercontent.com/28617444/125593182-18f69815-d2c2-4043-8693-8a04b596fd78.png">

- **해당 논문에서는 latent factor model을 model formulation, optimization technique를 수정하여 implicit feedback datasets에 적용하려 한다.**

## Our model
---

- $p_{ui}$ : 사용자 u가 i에 대한 preference 의미로 이진변수 도입 ( $r_{ui}$ > 0 이면 1 , $r_{ui}$ = 0 이면 0 )
    - $p_{ui}$ = 0 : low confidence와 연관 있음
- 실제로 preference한 아이템마다 다른 신뢰수준을 가지기에 행동을 한 관측치에 대한 설명이 부족함
    - 아이템에 preference가 0 : 아이템 존재를 모를수도, 구입할 수 있는 여력이 부족할수도 있음
    - 아이템 preference 1 : 이전 TV show보다 잠든 경우, 친구 선물일 경우
- **그러나 일반적으로 $r_{ui}$가 높아지면, 확실히 선호한 경향이 높음!**
    - 따라서, $c_{ui}$ 도입 : $p_{ui}$에서 confidence를 측정함을 의미

    ![Untitled 2](https://user-images.githubusercontent.com/28617444/125593187-61d88cd5-dac1-4d0c-887c-608658fc9bb4.png)

        - $r_{ui}$ 가 커질수록, 강한 선호지표를 나타내는 증가함수 (논문 실험 : 알파 = 40)
        - 효과
            1. 모든 user-item pair에 대한 minimal confidence 가질수 있음
            2. $p_{ui}$ = 1 인 경우, 증가함수
- 따라서, $p_{ui}$와 $c_{ui}$ 를 반영해서 user latent factor, item latent factor를 찾기 위한 loss function
![Untitled 3](https://user-images.githubusercontent.com/28617444/125593188-c88090f6-fcf8-4d5b-8e16-4a402dd38121.png)

    - 규제를 위한 람다는 cross-validation을 통해 정함

### 최적화 위한 방법

1. SGD
    - user 수 m , item 수 n 이 너무 많으면, SGD 같은 direct optimization technique를 방해한다.
        - 최적화할 수 있는 (n * k + m * k) 파라미터가 있기에, 원래의 rating 행렬의 차원이 높다면 좋은 선택이 아님!!
        - 이러한 상황에서의 효율적인 최적화 방법이 없을까?
2. ALS (Alternative Least Square)
    - user-factor 나 item-factor 중 하나를 고정시키고, 다른 하나를 최적화시키는 방법
        - $R - U*P^T$ 에서, P를 고정시키고, U에 대해서만 최적화시키면 선형회귀 문제이다!
            - $y - X*\beta$
            - 즉, OLS 는 unique하고 제일 작은 MSE를 보장하므로, cost function이 각 단계에서 감소하거나 변하지 않을 수 있기도 하지만 결코 증가하지는 않음
            - 두 단계를 번갈아 수행하면, 수렴할 때까지 cost function이 계속 감소하는 것을 보장

            ![Untitled 4](https://user-images.githubusercontent.com/28617444/125593190-9ed28db5-3b53-47c4-b228-2d369f9dcb83.png)

            - missing value 도입

            ![Untitled 5](https://user-images.githubusercontent.com/28617444/125593192-bb69be30-cf18-4ab6-b308-f7c548f3d909.png)

        - 이처럼 한 매트릭스를 고정시키면, quadratic (2차함수)가 되고 convex하기에 global minimum에 도달 가능

        - In 논문

        ![Untitled 6](https://user-images.githubusercontent.com/28617444/125593195-8a4b22ca-a41a-4156-ae1f-b0bc78ed0439.png) <br>
        ![Untitled 7](https://user-images.githubusercontent.com/28617444/125593199-444af36f-10b6-4e19-b976-60755ed681b1.png)

### Another Confidence level

- 앞에서는 linear 형태였지만, log 형태로도 나타낼 수 있다.

    <img src = "https://user-images.githubusercontent.com/28617444/125593201-8b9c556d-eb2d-4be9-aba9-7beba41803ac.png" width="50%">

1. $r_{ui}$ 를 변형하여 preference p_ui 와 confidence level c_ui 에 대한 해석이 가능
  - improve prediction accuracy - 실험 sec 6
2. 모든 n*m user-item combination을 다룰 수 있음


## Explaining recommendations
---
- 왜 그러한 상품을 추천하는지 설명할 수 있는가
    - neighborhood-based technique는 가능한 반면, latent factor model은 설명하기 어려움
    - 본 논문의 alternating least squares model은 설명하는 식을 제안함<br>
        user u의 item i에 대한 예측 preference : $y_i^Tx_u$

        ![image](https://user-images.githubusercontent.com/28617444/125594418-e8021298-15fc-4339-8390-531f7a1bbed7.png)
        - target item i에 대한 유사도 와 user u에 대한 relation로 설명가능하다는 의미

        - latent factor model ⇒ 과거 행동의 선형함수로써 preference를 예측하는 선형모델

## Experimental study
---
### Data description
- digital TV service
    - 300,000개의 set top boxes로부터 데이터 수집
    - 4주간 방영된 프로그램 17,000개
    - $r_{ui}$ : u가 프로그램 i를 몇 시간 시청했는지
    - 같은 프로그램의 중복 시청 반영 ⇒ non-zero 값 : 32 million (3천 2백만)

  #### 4주 - train set, 다음 1주 - test set

    - 짧은 기간은 예측결과를 악화시키므로
    - 긴 기간은 value가 추가가 안되므로 (계절성)

  #### 매주마다 같은 프로그램을 반복해서 보는 경향이 있음

    - 한번도 안보거나 최근에 안본게 추천시스템에서 값짐
    - 따라서, Training 기간에 시청한 프로그램은 test에서 삭제
        - 30분보다 작은 시청시간은 삭제
        - 따라서, test set에서 남은 non-zero 값 : 2 million(2백만)개

  #### 0부터 수백까지 매우 다양

    - 채널 훑어보기 : 0
    - 영화 또는 시리즈 에피소드 : 1,2,3
    - DVR : hundreds
    - **따라서, log scaling 적용**

  #### 동일한 채널에서 연속된 show 의 경우

    - 자는 경우가 대부분!!
    - down weight ( 실험 결과, a=2 , b=6 이 직관적 ⇒ 3번째 show 는 반으로 시간 줄고, 5번째는 99% 시간 줄어듬)

    ![image](https://user-images.githubusercontent.com/28617444/125594654-9871508a-16bf-4c9f-962e-8c0e999003c6.png)


### Evaluation methodology

- implicit data는 선호하지 않음을 나타낼 수가 없다! (앞에서 말한, 다양한 이유가 있으니)
- 또한, 추천해준 것에 대한 user의 반응을 알수도 없다!
    - 따라서, precision based metric이 아닌 recall-oriented measure가 적절할 것

    ![Untitled 11](https://user-images.githubusercontent.com/28617444/125593165-644d8bcf-413b-4449-91e0-a0208a6fbd5a.png)

  - precision : TP / (FP+TP) - 실제 Negative한 FP 의 경우, implicit data에서 정의가 되지 않는다
      - 실제로 추천한 영화중에 사용자가 선호하는 영화는 얼마나 되었나?
  - recall : TP / (FN + TP)
      - 실제 사용자가 선호하는 영화를 추천에서 얼마나 잘 맞췄나?

- positive 기준으로 ordered list를 생성
- $rank_{ui}$ = 0% : 가장 추천된 것
- $rank_{ui}$ = 100% : 가장 덜 추천된 것
    - rank_bar 를 계산한다.

    ![Untitled 12](https://user-images.githubusercontent.com/28617444/125593166-bc68b572-efaa-43be-88a9-224283e5aaf5.png)


### Evaluation results

#### 1) Rank

- factor의 수 $f$ : 10~200 실험
- 인기 순, neighborhood (item-item) 순보다 해당 모델이 훨씬 효과적임
    - popularity : 16.46% (개인에게 맞춤형이 아닌, 전체를 반영)
    - neighborhood based : 10.74%
    - 해당 모델 : 200 factor일 때, 8.35% 에 이름
    ![Untitled 13](https://user-images.githubusercontent.com/28617444/125593167-6d545437-0014-4728-bf73-afcd2dbc6da7.png)


#### raw observation ($r_{ui}$) ⇒ distinct preference-confidence pairs($p_{ui}, c_{ui}$)

1. regularized version of the dense SVD algorithm

![Untitled 14](https://user-images.githubusercontent.com/28617444/125593168-d2c2e137-d2ac-4ba4-8dfb-ccbabebe6197.png)

        - 규제 값 람다 가 없으면, 매우 poor 성능
        - 람다값 500 이 best recommendation
        - 그러나, neighborhood model 보다 poor 한 수준
        - 50 factor : rank_bar = 13.63% , 100 factor : rank_bar = 13.4%

2. factorize the derived binary preference values

![Untitled 15](https://user-images.githubusercontent.com/28617444/125593169-63f99dfb-48ae-4207-af2c-8eb035515361.png)

        - 람다 = 150
        - neighborhood model보다 약간 좋은 모델
        - 50 factor : rank_Bark = 10.72%, 100 factor : rank_bar = 10.49%

3. Our full model (confidence, preference)

![Untitled 16](https://user-images.githubusercontent.com/28617444/125593172-b260a7dd-5791-4597-adee-b171238e9933.png)

        - 50 ~ 100 factor : rank_bar = 8.93% ~ 8.56%

#### 2) Probability

- top X % 안에 유저가 추천받은 아이템을 시청할 누적 확률 분포
    - 해당 모델 : 상위 1% 추천 프로그램을 Test 시청시간의 27%를 담당

        즉, test set에서 추천을 긍정적으로 평가하여 show 를 시청할 확률이 높다.

        ![Untitled 17](https://user-images.githubusercontent.com/28617444/125593175-d7feadbf-a280-4b52-bc39-d02e0b10995a.png)

    - 점선 그래프: 해당 유저가 이전에 시청했던 프로그램을 삭제하지 않은 것인데, 다른 모델보다 선택할 확률이 매우 높았다.
        - 하지만 이는 이전에 봤던 데이터를 주는것이므로 실제 추천 시, 유용하지 않을 것

#### 3) Performance

- 15개의 bin으로 나누어서 "popularity" , "watching time"에 따른 측정한 rank 값이 얼마나 감소하는지를 평가

  ![Untitled 18](https://user-images.githubusercontent.com/28617444/125593176-2e3011a7-5f4a-4512-9c9b-0624ede46a42.png)

    - popular item : 추천받을 score 더 좋아짐
    - watching time : bin1(no watching history)인 경우 빼고는 의미가 없다!

#### 4) Utility

- 추천된 TV show와 해당되는 유사한 tv show

  ![Untitled 19](https://user-images.githubusercontent.com/28617444/125593179-b24110e5-c381-4e68-9123-f19bd0d2dc1c.png)

- Neighbor 기반 모델 : 주변 item 들을 통해 유저에게 추천의 이유를 알 수 있음
- 본 논문 : latent factor 기반 모델에서 이러한 neighbor 를 뽑을 수 있게 수식 변형 (기존과의 차이점)
- Top 5 가 35~40% 정도의 추천에 영향을 줌
    - 다른 많은 영화들이 latent factor 에 영향을 주기도 함

## Discussion 및 Conclusion
---
- preference , confidence level을 통해 implicit user observation을 정의한 것
- all user-item preference를 다루기 위해 algebraic structure을 이용한 것 ⇒ ALS
- preference 의 0가 의미하는 바를 해석하여 minimum 값을 준 것 ⇒ "no preference" assumption
- 추천시스템의 목적은 future user behavior를 맞추는 것이 아니라, user에게 item을 추천해주는 것! ⇒ 즉, re-watched shows를 제거함으로써 우리의 방법론을 평가하는 것이 이상적임!


## References
---

[1] [Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.](http://yifanhu.net/PUB/cf.pdf)
