# 22장 캐글로 시작하는 새로운 도전

캐글은 2010년 4월부터 지금가지 전 세계 데이터 과학자 15만 명 이상이 참가해 온 데이터 분석 경진대회입니다.
'데이터 사이언스를 스포츠처럼!'이라는 구호 아래 데이터 분석 기술을 스포츠와 같이 경쟁할 수 있게 만든 것이 특징입니다.
대회는 상금과 함께 상시 열리고 있으며, 각 경쟁마다 풀어야 할 과제와 평가 지표 그리고 실제 데이터가 주어집니다.
주어진 데이터를 사용해 정해진 시간 안에 가장 높은 정확도로 예측하는 것이 목표이지요.
분석 결과를 업로드하면 보통 몇 분 안에 채점이 끝나며, 평가 지표에 근거해 참가자 간의 순위가 매겨집니다.
실제 데이터를 사용해 다양한 기술을 구현하므로 자신의 데이터 과학 수준을 확인할 수 있을 뿐만 아니라 최신 기술과 트렌드를 배울 수 있는 기회를 제공합니다.
이 장에서 우리는 캐글에 가입하는 방법과 캐글에 예측된 결과를 업로드하는 방법을 배울 것입니다.

캐글에 참여하는 순서는 다음과 같습니다.
1. 캐글 가입 및 대회 선택하기
2. 데이터 획득
3. 학습하기
4. 결과 제출 및 업데이트
5. 최종 예측 결과

## 1 캐글 가입 및 대회 선택하기

먼저 캐글 웹 사이트에 방문해 회원 가입을 합니다. 
구글 계정이 있으면 간단히 회원에 가입할 수 있습니다.

![image](https://user-images.githubusercontent.com/52357235/179709931-fc797c4e-8dce-4038-83bd-9f50bf9919d3.png)

가입이 완료되면, 캐글에 공지된 대회 중 참가할 만한 대회를 선택합니다. 
메인 화면에서 Competitions를 클릭하면 현재 진행 중인 경진대회의 목록이 보입니다.

우리는 스터디를 목적으로 하므로 캐글에서 누구나 테스트할 수 있게끔 준비한 House Prices - Advanced Regression Techniques를 클릭합니다.

## 2 데이터 획득하기

해당 경진대회에 접속을 완료하면 대회에 대한 내용을 숙지하고 Data를 클릭해 데이터에 접근합니다.

데이터 화면이 나오면 I understand and agree를 클릭해 데이터를 내려받을 준비를 합니다.

데이터 화면이 바뀌고 해당 데이터에 대한 설명이 나옵니다. Download All을 클릭합니다.

내려받은 데이터를 확인해 보겠습니다.

![image](https://user-images.githubusercontent.com/52357235/179712705-2f6aa6dc-6dec-404e-a3d7-896163ec3c52.png)

data_description.txt 파일은 내려받은 데이터의 각 속성이 무엇을 의미하는지 설명하고 있습니다.
train.csv 파일은 집 값과 해당 집이 어떤 속성을 가졌는지 정리된 파일입니다.

test.cvs 파일은 train.csv 파일을 이용해 학습한 결과를 테스트하기 위한 데이터입니다.
train.cvs 파일과 모든 항목이 같지만 맨 마지막 집 값(SalePrice) 항목만 빠져 있습니다.
이 항목을 예측하는 것이 우리의 과제입니다.
sample_submission.csv 파일은 Id와 SalePrice 두 개의 열만 존재하는 파일입니다.
각 Id별로 우리가 예측한 SalePrice를 채워 넣어 캐글에 업로드하면 됩니다.

## 3 학습하기

데이터를 확인했으면 이제 딥러닝 또는 머신 러닝 기법을 활용해 모델을 만들고 학습을 시작하면 됩니다.

먼저 필요한 라이브러리를 불러옵니다.
케라스의 load_model과 판다스를 불러오겠습니다.

``` python
from tensorflow.keras.models import load_model
import pandas as pd
```

캐글에서 배포하는 house_test.csv 파일은 data 폴더에 이미 저장되어 있습니다.
해당 테스트셋을 불러오겠습니다.

``` python
kaggle_test = pd.read_csv("./data/house_test.csv")
```

테스트셋의 속성은 학습셋과 동일한 상태로 변형되어야 해당 모델을 적용할 수 있습니다.
이를 위해 학습셋과 동일하게 전처리되어야 합니다.
먼저 카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 주겠습니다.

``` python
kaggle_test = pd.get_dummies(kaggle_test)
```

그리고 결측치를 전체 칼럼의 평균으로 대체해 채워 줍니다.

``` python
kaggle_test = kaggle_test.fillna(kaggle_test.mean())
```

이제 학습에 사용된 열을 K_test로 저장합니다.

``` python
cols_kaggle = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
K_test = kaggle_test[cols_kaggle]
```

앞서 15장에서 만든 모델을 불러옵니다.

``` python
model = load_model("./data/model/Ch15-house.hdf5")
```

model.predict()를 이용해 불러온 모델에 조금 전 K_test를 적용하고 예측 값을 만들어 봅니다.

``` python
ids = [] # ID와 예측 값이 들어갈 빈 리스트를 만듭니다.
Y_prediction = model.predict(K_test).flatten()
for i in range(len(K_test)):
  id = kaggle_test['Id'][i]
  prediction = Y_prediction[i]
  ids.append([id, prediction])
```

테스트 결과의 저장 환경을 설정합니다.
앞서 만든 내용과 중복되지 않도록 현재 시간을 이용해 파일명을 만들어 저장하겠습니다.
파일은 별도 폴더에 저장되도록 하겠습니다.

``` python
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
filename = str(timestr) # 파일명을 연월일-시분초로 정합니다.
outdir = './' # 파일이 저장될 위치를 지정합니다.
```

앞서 만들어진 실행 번호와 예측 값을 새로운 데이터 프레임에 넣고 이를 csv 파일로 저장합니다.

``` python
df = pd.DataFrame(ids, columns=["Id", "SalePrice"])
df.to_csv(str(outdir + filename + '_submission.csv'), index=False)
```

[실습1 캐글에 제출할 결과 만들기](https://github.com/zzzangmans1/DeepLearning/blob/main/22/22.py)

이코드를 실행해 구글 코랩 폴더에 (연도)(월)(일)-(시)(분)(초)_submission.csv 파일이 만들어 졌다면 결과를 캐글에 제출할 준비가 되었습니다.

## 4 결과 제출하기

다시 경진대회 웹 페이지로 돌아가서 이번에는 Submit Predictions를 클릭합니다.

