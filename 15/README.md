 # 15장 실제 데이터로 만들어 보는 모델

지금까지 한 실습은 참 또는 거짓을 맞히거나 여러 개의 보기 중 하나를 예측하는 분류 문제였습니다.
그런데 이번에는 수치를 예측하는 문제입니다. 준비된 데이터는 아이오와주 에임스 지역에서 2006년부터 2010년까지 거래된 실제 부동산 판매 기록입니다. 주거 유형, 차고, 자재 및 환경에 관한 80개의 서로 다른 속성을 이용해 집의 가격을 예측해 볼 예정인데 오랜 시간 사람이 일일이 기록하다 보니 빠진 부분도 많고, 집에 따라 어떤 항목은 범위에서 너무 벗어나 있기도 하며, 또 가격과는 관계가 없는 정보가 포함되어 있기도 합니다. 
현장에서 만나게 되는 이런 류의 데이터를 어떻게 다루어야 하는지 이 장에서 학습해 보겠습니다.

## 1  데이터 파악하기


|Source|Description|
|--|--|
|import pandas as pd|필요한 api 를 불러옵니다.|
|!git clone https://github.com/taehojo/data.git|깃허브에 준비된 데이터를 가져옵니다.|
|df = pd.read_csv("./data/house_train.csv")|집 값 데이터를 불러옵니다.|
|df|출력합니다.|


![image](https://user-images.githubusercontent.com/52357235/177986304-5fec8406-2b05-43ea-972b-bdc1f6c95422.png)


총 80개의 속성으로 이루어져 있고 마지막 열이 우리의 타깃인 집 값(SalePrice)입니다.
모두 1,460개의 샘플이 들어 있습니다.
이제 각 데이터가 어떤 유형으로 되어 있는지 알아보겠습니다.

|Source|Description|
|--|--|
|df.dtypes|각 데이터가 어떤 유형으로 되어 있는지 알아본다|

![image](https://user-images.githubusercontent.com/52357235/177986573-7cf901a2-64f1-4fee-a8ee-65e25210327f.png)


---

## 2 결측치, 카테고리 변수 처리하기

앞 장에서 다루었던 데이터와 차이점은 아직 전처리가 끝나지 않은 상태의 데이터라 측정값이 없는 결측치가 있다는 것입니다. 결측치가 있는지 알아보는 함수는 isnull()입니다. 결측치가 모두 몇 개인지 세어 가장 많은 것부터 순서대로 나여한 후 처음 20개만 출력하는 코드는 다음과 같습니다.

|Source|Description|
|--|--|
|df.isnull().sum().sort_values(ascending=False).head(20)|결측치가 모두 몇 개인지 세어 가장 많은 것부터 순서대로 나여한 후 처음 20개만 출력|

![image](https://user-images.githubusercontent.com/52357235/177987762-d751cfee-3d8b-4e65-9c05-4fb69f4d3bc3.png)

이제 모델을 만들기 위해 데이터를 전처리하겠습니다. 먼저 12.3절에서 소개되었던 판다스의 get_dummies() 함수를 이용해 카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줍니다.
|Source|Description|
|--|--|
|df = pd.get_dummies(df)|카테고리형 변수 0과 1로 이루어진 변수로 바꾸어 줍니다.|

그리고 결측치를 채워 줍니다. 결측치를 채워 주는 함수는 판다스의 fillna()입니다. 괄호 안에 df.mean()을 넣어 주면 평균값으로 채워 줍니다.

|Source|Description|
|--|--|
|df = df.fillna(df.mean())|결측치를 채워 주기 위해 평균값을 넣어 결측치를 채워 줍니다.|

![image](https://user-images.githubusercontent.com/52357235/177988327-32ad1f3c-ef94-4531-b346-9502242ce287.png)

결측치는 보이지 않으며 카테고리형 변수를 모두 원-핫 인코딩 처리하므로 전체 열이 81개에서 290개로 늘었습니다.

---

## 3 속성별 관련도 추출하기

이 중에서 우리에게 필요한 정보를 추출해 보겠습니다.

|Source|Description|
|--|--|
|df_corr = df.corr()|먼저 데이터 사이의 상관관계를 df_coor 변수에 저장합니다. |
|df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)|집 값과 관련이 큰 것부터 순서대로 정렬해 df_corr_sort 변수에 저장합니다. |
|df_corr_sort['SalePrice'].head(10)|집값과 관련도가 가장 큰 열 개의 속성들을 출력합니다.|

---
