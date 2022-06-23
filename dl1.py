# 텐서플로 라이브러리 안에 있는 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터를 다루는 데 필요한 라이브러리를 불러옵니다.
import numpy as np

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 준비된 수술 환자 데이터를 불러옵니다.
Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")

X = Data_set[:,0:16] # 환자의 진찰 기록을 X로 지정합니다.
y = Data_set[:,16] # 수술 1년 후 사망/생존 여부를 y로 지정합니다.

# 딥러닝 모델의 구조를 결정합니다.
# 딥러닝의 모델을 설정하고 구동하는 부분은 모두 model이라는 함수를 선언하며 시작됩니다.
model = Sequential() # Sequential - 딥러닝의 구조를 짜고 층을 설정하는 부분
# add 는 새로운 층을 만드는 부분
model.add(Dense(30, input_dim=16, activation='relu')) # 그 외 층은 모두 입력층 + 은닉층
# 쉽게 설명해 30개의 노드에 입력값 16개를 보낸다는 의미이다.
model.add(Dense(1, activation='sigmoid')) # 맨 마지막 층은 출력층
# 출력값을 하나로 정해서 보여 주어야 하므로 출력층의 노드 수는 한 개이다. 그리고 활성화 함수를 거쳐 최종 출력 값으로 나와야 합니다. 그 함수는 sigmoid 함수를 사용합니다.

# 딥러닝 모델을 실행합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # compile - 앞에서 정한 모델을 컴퓨터가 알아들을 수 있게끔 컴파일하는 부분
# 먼저 어떤 오차 함수를 사용할지 정해야 합니다. 손실 함수에는 두 가지 종류가 있다. 선형회귀 - 평균 제곱 오차, 로지스틱 회귀 - 교차 엔트로피 오차
# 생존율 예측은 생존과 사망, 둘 중 하나를 예측하므로 교차 엔트로피 오차 함수를 적용해서 binary_crossentropy를 선택
# 여럿 중 하나 예측은 - categorical_crossentropy
# 가장 많이 사용하는 optimizer = adam 
# metrics - 함수는 모델이 컴파일될 때 모델 수행의 결과를 나타내게끔 설정하는 부분 
# accuracy라고 설정한 것은 학습셋에 대한 정확도에 기반해 결과를 출력하라는 의미

history = model.fit(X, y, epochs=5, batch_size=500) # fit - 모델을 실제 실행하는 부분
# epochs 는 각 샘플이 처음부터 끝까지 다섯 번 재사용될 때까지 실행을 반복하라는 의미
# batch_size 는 샘플을 한 번에 몇 개씩 처리할지 정하는 부분
# batch_size 는 자신의 컴퓨터 메모리가 감당할 만큼의 batch_size를 찾아 설정해주는 것이 좋다.
