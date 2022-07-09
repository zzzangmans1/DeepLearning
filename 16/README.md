# 16장 이미지 인식의 꽃, 컨볼루션 신경망(CNN)

급히 전달받은 노트에 숫자가 적혀 있습니다. 뭐라고 썼는지 읽기에 그리 어렵지 않습니다. 일반적인 사람에게 이 사진에 나온 숫자를 읽어 보라고 하면 대부분 '504192'라고 읽겠지요.
그런데 컴퓨터에 이 글씨를 읽게 하고 이 글씨가 어떤 의미인지 알게 하는 과정은 쉽지 않습니다. 사람이 볼 때는 쉽게 알 수 있는 글씨라 해도 숫자 5는 어떤 특징을 가졌고, 숫자 9는 
6과 어떻게 다른지 기계가 스스로 파악해 정확하게 읽고 판단하게 만드는 것은 머신 러닝의 오랜 진입 과제였습니다.
MNIST 데이터셋은 미국 국립표준기술원(NIST)이 고등학생과 인구조사국 직원 등이 쓴 손글씨를 이용해 만든 데이터로 구성되어 있습니다.
7만 개의 글자 이미지에 각각 0부터 9까지 이름표를 붙인 데이터셋으로, 머신 러닝을 배우는 사람이라면 자신의 알고리즘과 다른 알고리즘의 성과를 비교해 보고자 한 번씩 도전해 보는
유명한 데이터 중 하나이지요.

## 1 이미지를 인식하는 원리

MNIST 데이터는 텐서플로의 케라스 API를 이용해 간단히 불러올 수 있습니다. 함수를 이용해서 사용할 데이터를 불러옵니다.

|source|description|
|--|--|
|from tensorflow.keras.datasets import mnist|MNIST 임폴트|

이때 불러온 이미지 데이터를 X로, 이 이미지에 0~9를 붙인 이름표를 y로 구분해 명명하겠습니다. 또한, 7만 개 중 학습에 사용될 부분은 train으로, 
테스트에 사용될 부분은 test라는 이름으로 불러오겠습니다.

- 학습에 사용될 부분: X_train, y_train
- 테스트에 사용될 부분: X_test, y_test

|source|description|
|--|--|
|(X_train, y_train), (X_test, y_test) = mnist.load_data()| MNIST 데이터는 7만 개 이미지 중 6만 개를 학습용, 1만 개를 테스트용으로 미리 구분해 놓고 있습니다.|
|print("학습셋 이미지 수: %d개" % (X_train.shape[0]))|학습셋 이미지 수 출력|
|print("테스트셋 이미지 수: %d개" % (X_test.shape[0]))|테스트셋 이미지 수 출력|

![image](https://user-images.githubusercontent.com/52357235/178097945-3d8676da-9bda-43e0-945e-5449a3deeedb.png)

불러온 이미지 중 한 개만 다시 불러와 보겠습니다.
그리고 할 수 있습니다. 모든 이미지가 X_train에 저장되어 있으므로을 지정해  출력되게 합니다.
|source|desciption|
|--|--|
|import matplotlib.pyplot as plt|맷플롭립 라이브러리를 불러옵니다.|
|plt.imshow(X_train[0], cmap='Greys'| X_train[0]으로 첫 번째 이미지를, cmap='Greys' 옵션을 지정해 흑백으로 imshow() 함수를 이용해 이미지를 출력|
|plt.show()|그래프로 출력|

![image](https://user-images.githubusercontent.com/52357235/178098852-6483e264-b767-465b-b144-debe7d98fa61.png)

이 이미지를 컴퓨터는 어떻게 인식할까요?
이 이미지는 가로 28 X 세로 28 = 총 784개의 픽셀로 이루어져 있습니다. 각 픽셀은 밝기정도에 따라 0부터 255까지 등급을 매깁니다.
흰색 배경이 0이라면 글씨가 들어간 곳은 1~255의 숫자 중 하나로 채워져 긴 행렬로 이루어진 하나의 집합으로 변환됩니다.

|source|desciption|
|--|--|
|for x in X_train[0]:|X_train[0]의 값을 x에 넣습니다.|
|for i in x:|x의 값을 i에 넣습니다.|
|sys.stdout.write("%-3s" % i)|i값을 오른쪽에 공백3칸을 주고 출력합니다.|
|sys.stdout.write('\n')|한줄 출력하고 개행문자를 출력합니다.|

바로 이렇게 이미지는 다시 숫자의 집합으로 바뀌어 학습셋으로 사용됩니다. 
우리가 앞서 배운 여러 예제와 마찬가지로 속성을 담은 데이터를 딥러닝에 집어넣고 클래스를 예측하는 문제로 전환시키는 것이지요.
28 X 28 = 784개의 속성을 이용해 0~9의 클래스 열 개 중 하나를 맞히는 문제가 됩니다.
이제 주어진 가로 28, 세로 28의 2차원 배열을 784개의 1차원 배열로 바꾸어 주어야 합니다.
이를 위해 reshape() 함수를 사용합니다.
reshape(총 샘플 수, 1차원 속성의 개수) 형식으로 지정합니다. 
총 샘플 수는 앞서 사용한 X_train.shape[0]을 이용하고, 1차원 속성의 개수는 이미 살펴본 대로 784개입니다.

|source|description|
|--|--|
|X_train = X_train.reshape(X_train.shape[0], 784)|X_train.shape[0] 의 2차원 배열 값을 784개의 1차원 배열로 reshape 함수를 사용해 바꾸어 줍니다.|

케라스는 데이터를 0에서 1 사이의 값으로 변환한 후 구동할 때 최적의 성능을 보입니다. 따라서 현재 0~255 사이의 값으로 이루어진 값을 0~1 사이의 값으로 바꾸어야 합니다.
바꾸는 방법은 각 값을 255로 나누는 것입니다. 이렇게 데이터의 폭이 클 때 적절한 값으로 분산의 정도를 바꾸는 과정을 데이터 **정규화**라고 합니다.

|source|description|
|--|--|
|X_train = X_train.astype('float64')|데이터 타입을 float64로 변경합니다.|
|X_train = X_train / 255|값을 255로 나누어 줍니다.|
|print("class : %d " % (y_train[0])|실제로 이 숫자의 레이블이 어떤지 y_train[0]을 출력하겠습니다.|

![image](https://user-images.githubusercontent.com/52357235/178099866-f5c2a185-e28e-4b1b-ae65-8fa2fe79f739.png)

지금 우리가 열어 본 이미지의 클래스는 [5] 였습니다. 이를 [0,0,0,0,0,1,0,0,0,0,0]으로 바꾸어야 합니다.
이를 가능하게 해 주는 함수가 바로 np_utils.to_categorical() 함수입니다.
to_categorical(클래스, 클래스의 개수) 형식으로 지정합니다.

|source|description|
|--|--|
|y_train = to_categorical(y_train, 10)|y_train을 리스트 형식으로 변경합니다.|
|y_test = to_categorical(y_test, 10)|y_test를 리스트 형식으로 변경합니다.|


![image](https://user-images.githubusercontent.com/52357235/178099780-a6431302-4786-4983-aa9d-0c6ec5831026.png)


[실습1 MNIST 손글씨 인식하기: 데이터 전처리](https://github.com/zzzangmans1/DeepLearning/blob/main/16/16_1.py)

![image](https://user-images.githubusercontent.com/52357235/178100022-2ff33e70-364a-4071-a6ec-9f1572e1e1d5.png)

## 2 딥러닝 기본 프레임 만들기

이제 불러온 데이터를 실행할 차례입니다. 
총 6만 개의 학습셋과 1만 개의 테스트셋을 불러와 속성 값을 지닌 X, 클래스 값을 지닌 y로 구분하는 작업을 다시 한 번 정리하면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/52357235/178100095-879a74af-313e-4d2d-9cf5-e277b0fac11e.png)

이제 딥러닝을 실행하고자 프레임을 설정합니다. 총 784개의 속성이 있고 열 개의 클래스가 있습니다. 
따라서 다음과 같이 딥러닝 프레임을 만들 수 있습니다.

![image](https://user-images.githubusercontent.com/52357235/178100218-7e62fbe6-25b2-4c24-a06d-07825e50d4a2.png)

입력값(input_dim)이 784개, 은닉층이 512개, 출력이 열 개인 모델입니다. 
활성화 함수로 은닉층에서는 relu를, 출력층에서는 softmax를 사용했습니다.
그리고 딥러닝 실행 환경을 위해 오차 함수로 categorical_crossentropy, 최적화 함수로 adam을 사용하겠습니다.

![image](https://user-images.githubusercontent.com/52357235/178100277-8bbd0b11-173c-4d0c-87d9-bde696f73bdf.png)

모델 실행에 앞서 먼저 성과를 저장하고, 모델의 최적화 단계에서는 학습을 자동 중단하게끔 설정하겠습니다.
열 번 이상 모델 성능이 향상되지 않으면 자동으로 학습을 중단합니다.

![image](https://user-images.githubusercontent.com/52357235/178100701-e3b318f7-d571-4571-9f84-8f0a83cd1f9a.png)

샘플 200개를 모두 30번 실행하게끔 설정합니다.
그리고 테스트셋으로 최종 모델의 성과를 측정해 그 값을 출력합니다.

![image](https://user-images.githubusercontent.com/52357235/178101214-51ef223f-3d7e-408b-bd01-76af957a3b3a.png)

실행 결과를 그래프로 표현해 보겠습니다. 
이번에는 학습셋의 정확도 대신 학습셋의 오차를 그래프로 표현하겠습니다. 학습셋의 오차는 1에서 학습셋의 정확도를 뺀 값입니다. 
좀 더 세밀한 변화를 볼 수 있게 학습셋의 오차와 테스트셋의 오차를 그래프 하나로 나타내겠습니다.

![image](https://user-images.githubusercontent.com/52357235/178101468-6f5eac59-450c-4d1f-909b-07321953272a.png)

[실습1 MNIST 손글씨 인식하기: 기본프레임](https://github.com/zzzangmans1/DeepLearning/blob/main/16/16_2.py)

![image](https://user-images.githubusercontent.com/52357235/178101747-e00e82f6-cb5f-486c-9e8b-f7f3c0ace56f.png)

## 3 컨볼루션 신경망(CNN)

컨볼루션 신경망은 입력된 이미지에서 다시 한 번 특징을 추출하기 위해 커널(슬라이딩 윈도)을 도입하는 기법입니다.

케라스에서 컨볼루션 층을 추가하는 함수는 Conv2D()입니다. 
다음과 같이 컨볼루션 층을 적용해 MNIST 손글씨 인식률을 높여 봅시다.

![image](https://user-images.githubusercontent.com/52357235/178102082-e38a0c49-b9db-4bf1-bf84-77d40cc6550f.png)

이어서 컨볼루션 층을 하나 더 추가해 보겠습니다.

![image](https://user-images.githubusercontent.com/52357235/178102148-2c4fc3cb-0a69-443d-88b7-15a8740ca146.png)

## 4 맥스 풀링, 드롭아웃, 플래튼

앞서 구현한 컨볼루션 층을 통해 이미지 특징을 도출했습니다.
하지만 그 결과가 여전히 크고 복잡하면 이를 다시 한 번 축소해야 합니다.
이 과정을 풀링 또는 서브 샘플링이라고 합니다.
이러한 풀링 기법에는 정해진 구역 안에서 최댓값을 뽑아내는 맥스 풀링과 평균값을 뽑아내는 평균 풀링 등이 있습니다.

이 중 보편적으로 사용되는 것은 맥스 풀링입니다.
맥스 풀링의 크기만큼 적용하여 구역을 나눕니다.
그리고 각 구역에서 가장 큰 값을 추출합니다.

![image](https://user-images.githubusercontent.com/52357235/178102308-98f5a419-847c-4a20-ad2f-0974f19c4032.png)

- 드롭아웃, 플래튼
노드가 많아지거나 층이 많아진다고 해서 학습이 무조건 좋아지는 것이 아니라는 점을 과적합 의미를 공부하며 배웠습니다. 
딥러닝에서 학습을 진행할 때 가장 중요한 것은 과적합을 얼마나 효과적으로 피해 가는지에 달려 있다고 해도 과언이 아닙니다.
따라서 그동안 이러한 과정을 도와주는 기법이 연구되어 왔습니다.
그 중 간단하지만 효과가 큰 기법이 바로 **드롭아웃(drop out)** 기법입니다. 
드롭아웃은 은닉층에 배치된 노드 중 일부를 임의로 꺼 주는 것입니다.

예를 들어 25%의 노드를 끄려면 다음과 같이 코드를 작성하면 됩니다.

![image](https://user-images.githubusercontent.com/52357235/178102406-7e4b4f39-94a7-4e2e-bf0b-61b94145574a.png)

이제 이러한 과정을 지나 다시 앞에서 Dense() 함수를 이용해 만들었던 기본 층에 연결하겠습니다.
이때 주의할 점은 컨볼루션 층이나 맥스 풀링은 주어진 이미지를 2차원 배열인 채로 다룬다는 것입니다.
이를 1차원 배열로 바꾸어 주어야 활성화 함수가 있는 층에서 사용할 수 있습니다. 
따라서 Flatten() 함수를 사용해 2차원 배열을 1차원으로 바꾸어 줍니다.

![image](https://user-images.githubusercontent.com/52357235/178102441-fb001cdb-e80d-4d34-ba0e-d5cf39ffe9bb.png)

## 5 컨볼루션 신경망 실행하기

[실습1 MNIST 손글씨 인식하기: 컨볼루션 신경망 적용](https://github.com/zzzangmans1/DeepLearning/blob/main/16/16_3.py)

![image](https://user-images.githubusercontent.com/52357235/178103516-616323bf-c88b-4bfd-b0b7-946f1a0c354f.png)
