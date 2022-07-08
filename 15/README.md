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

## 3 속성별 관련도 추출하기

이 중에서 우리에게 필요한 정보를 추출해 보겠습니다.
|Source|Description|
|--|--|
|df_corr = df.corr()|먼저 데이터 사이의 상관관계를 df_coor 변수에 저장합니다. |
|df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)|집 값과 관련이 큰 것부터 순서대로 정렬해 df_corr_sort 변수에 저장합니다. |
|df_corr_sort['SalePrice'].head(10)|집값과 관련도가 가장 큰 열 개의 속성들을 출력합니다.|

![image](https://user-images.githubusercontent.com/52357235/177990373-c18d6ada-f605-4e86-a2d6-7680c6546b8d.png)

추출된 속성들과 집 값의 관련도를 시각적으로 확인하기 위해 상관도 그래프를 그려 보겠습니다.

|Source|Description|
|--|--|
|import matplotlib.pyplot as plt|plt api를 사용하기 위해 라이브러리 호출|
|import seaborn as sns|sns api를 사용하기 위해 라이브러리 호출|
|cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','GarageArea','TotalBsmtSF']|속성들을 cols 변수에 리스트 형식으로 넣습니다.|
|sns.pairplot(df[cols])|데이터에 들어 있는 각 컬럼(열)들의 모든 상관 관계를 출력|
|plt.show()|그래프 출력|

![image](https://user-images.githubusercontent.com/52357235/177991264-90ddee37-96b9-4917-ab6b-e2761dcd9db3.png)

## 4 주택 가격 예측 모델

이제 앞서 구한 중요 속성을 이용해 학습셋과 테스트셋을 만들어 보겠습니다. 집 값을 y로 나머지 열을 X_train_pre로 저장한 후 전체의 80%를 학습셋으로, 20%를 테스트셋으로 저장합니다.

|Source|Description|
|--|--|
|cols_train = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']|중요한 속성만 따로 리스트로 저장|
|X_train_pre = df[cols_train]|중요한 속성들을 X_train_pre 학습셋으로 저장|
|y = df['SalePrice'].values|SalePrice를 y로 저장|
|X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)|80%를 학습셋으로 20%를 테스트셋으로 저장|

모델의 구조와 실행 옵션을 설정합니다. 입력될 속성의 개수는 X_train.shape[1]로 지정해 자동으로 세도록 했습니다.

|Source|Description|
|--|--|
|model = Sequential()|Sequential 모델 오브젝트를 model 변수에 저장합니다.|
|model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))|모델의 층을 추가하고 출력은 10, 입력은 X_train.shape[1]로 자동, 활성함수는 relu|
|model.add(Dense(30, activation='relu'))|모델의 층을 추가하고 출력은 30, 활성함수는 relu|
|model.add(Dense(40, activation='relu'))|모델의 층을 추가하고 출력은 40, 활성함수는 relu|
|model.add(Dense(1))|출력은 1입니다. 출력층입니다.|
|model.summary()|모델의 구조를 요약 출력해줍니다.|

실행에서 달라진 점은 손실 함수입니다. 선형 회귀이므로 평균 제곱 오차(mean_squared_error)를 적습니다.
|Source|Description|
|--|--|
|model.compile(optimizer='adam', loss='mean_squared_error')|optimizer 최적알고리즘은 adam, 모델이 최적화에 사용하는 손실함수는 평균 제곱 오차함수|

20번 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다. 저장된 모델 이름을 'Ch15-house.hdf5로 정하겠습니다. 
모델은 차후 '22장. 캐글로 시작하는 새로운 도전'에서 다시 사용됩니다(검증셋을 추가하고 싶을 경우 앞서와 마찬가지로 학습셋, 검증셋, 테스트셋의 비율을 각각 60%, 20%, 20%로 정하면 됩니다). 

|Source|Description|
|--|--|
|early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)|monitor - 학습 조기종료를 위해 관찰하는 변수, patience - 몇번 이상 결과가 향상되지 않으면 중단|
|modelpath = "./data/model/Ch15-house.hdf5"|모델이 저장될 경로는 "./data/model/Ch15-house.hdf5"|
|checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True|filepath - 모델을 저장할 경로, monitor - 모델을 저장할 때 기준이 되는 값, verbose - 1일 경우 모델이 저장 될 때 저장되었습니다. 표시, 0일 경우 화면에 표시되는 것 없이 바로 저장, save_best_only - True 인 경우, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델이 저장, Flase 인 경우, 매 에폭마다 모델이 filepath{epoch{으로 저장|
|history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])|X 값과 y값을 넣고, validation_split - 검증셋 옵션, epochs - 몇번 반복, batch_size - batch_size 만큼 쪼개서 학습, callback - callback 함수들 지정|

# 실습 1주택 가격 예측하기

![image](https://user-images.githubusercontent.com/52357235/177999176-a2413686-7415-49f6-9c62-8485f8177d75.png)

