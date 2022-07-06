from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 

# pandas 라이브러리를 불러옵니다.
import pandas as pd

# k겹 교차 검증 - 데이터셋을 여러 개로 나누어 하나씩 테스트셋으로 사용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법입니다.
# EX) 데이터셋 = 1, 2, 3, 4, 5  테스트 셋 - ()
# 1 , 2,  3, 4, (5) = 결과 1
# 1, 2, 3, (4), 5 = 결과 2
# 1, 2, (3), 4, 5 = 결과 3
# 1, (2), 3, 4, 5 = 결과 4
# (1), 2, 3, 4, 5 = 결과 5 
# 결과를 다 더해 평균을 구한다.

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 광물 데이터를 불러옵니다.
df = pd.read_csv('./data/sonar3.csv', header=None)

df.head() # 첫 다섯 줄을 봅니다.

# 전체가 61개의 열로되어 있고, 마지막 열이 광물의 종류를 나타냅니다. 일반 암석일 경우 0, 광석일 경우 1로 표시
# 첫 번째 열부터 60번째 열까지는 음과 주파수의 에너지를 0에서 1사이의 숫자로 표시하고 있습니다.

df[60].value_counts() # 일반 암석과 광석이 각각 몇 개나 포함되어 있는지 알아보는 함수

X = df.iloc[:,0:60] # 0~59 번째 데이터는 X에 저장
y = df.iloc[:,60] # 60번째 데이터는 y에 저장

# 몇 겹으로 나눌 것인지 정합니다.
k = 5

# KFold 함수를 불러옵니다. 분할하기 전에 샘플이 치우치지 않도로 섞어 줍니다.
kfold = KFold(n_splits=k, shuffle=True)

# 정확도가 채워질 빈 리스트를 준비합니다.
acc_score = []

def model_fn():
  # 모델을 설정합니다.
  model = Sequential()
  model.add(Dense(24, input_dim=60, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  return model

# k겹 교차 검증을 이용해 k번의 학습을 실행합니다.
# for 문에 의해 k번 반복합니다.
# split()에 의해 k개의 학습셋, 테스트셋으로 분리됩니다.
for train_index, test_index in kfold.split(X):
  X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model = model_fn()
  # 모델을 컴파일합니다.
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # 모델을 실행합니다.
  history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

  accuracy = model.evaluate(X_test, y_test)[1] # 정확도를 구합니다.
  acc_score.append(accuracy)                   # 정확도 리스트에 저장합니다.

# k 번 실시된 정확도의 평균을 구합니다.
avg_acc_score = sum(acc_score) / k
# 결과를 출력합니다.
print('정확도: ', acc_score)
print('정확도 평균: ',avg_acc_score)
# 모델 이름과 저장할 위치를 함께 지정합니다.
model.save('./data/model/my_model.hdf5') # hdf5 파일 포맷은 주로 과학 기술 데이터 작업에서 사용되는데, 크고 복잡한 데이터를 저장하는 데 사용됩니다.
