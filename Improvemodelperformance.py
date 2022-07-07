from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd

# EarlyStopping() 함수와 ModelCheckpoint() 함수를 통해 모델 성능 향상시키는 예제

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 와인 데이터를 불러옵니다.
df = pd.read_csv('./data/wine.csv', header=None)

# 데이터를 미리 보겠습니다.
# df

# 샘플이 전체 6,497개 있습니다.
# 모두 속성이 12개 기록되어 있고 13번째 열에 클래스가 준비되어 있습니다.
# 각 속성에 대한 정보는 다음과 같습니다.
# 0 주석산 농도, 1 아세트산 농도, 2 구연산 농도, 3 잔류 당분 농도, 4 염화나트륨 농도, 5 유리 아황산 농도, 6 총 아황산 농도
# 7 밀도, 8 pH, 9 황산칼륨 농도, 10 알코올 도수, 11 와인의 맛(0~10등급), 12 레드와인:1, 화이트와인:0

# 0~11번째 열에 해당하는 속성 12개를 X로, 13번째 열을 y로 지정
X = df.iloc[:,0:12]
y = df.iloc[:,12]

# 이전에는 학습셋과 테스트셋을 나누는 방법을 알아보았습니다.
# 이 장에서는 여기에 검증셋을 더해 보겠습니다.

# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 모델 업데이트하기
# 에포크가 50번이면 순전파와 역전파를 50번 실시한다는 뜻입니다.
# 학습을 많이 반복한다고 해서 모델 성능이 지속적을 좋아지는 것은 아닙니다.

# 에포크마다 모델의 정확도를 함께 기록하면서 저장하는 방법
modelpath = "./data/model/all/Ch14-4-bestmodel.hdf5"

# 학습 중인 모델을 저장하는 함수는 케라스 api의 ModelCheckpoint() 입니다.
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 학습이 언제 자동 중단될지 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20) # 20번 오차가 변화 없을때 종료

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)



# 모델을 실행합니다.
history = model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1, callbacks=[early_stopping_callback, checkpointer]) #  0.8 * 0.25 = 0.2

# 테스트 결과를 출력합니다.
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
