from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

# 실습1 MNIST 손글씨 인식하기: 컨볼루션 신경망 적용

# MNIST 데이터셋을 불러와 학습셋과 테스트셋으로 저장합니다.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 차원 변환 과정을 실습해 봅시다.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],28, 28, 1).astype('float32') / 255

# 바이너리화 과정을 실습해 봅니다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
# Conv2D 
# 첫 번째 인자: 커널을 몇 개 적용할지 정합니다. 여기서는 32개의 커널 적용
# 두 번째 인자: 커널의 크기를 정합니다. (행, 열) 형식으로 정하며 여기서는 3 X 3 크기의 커널 사용
# 세 번째 인자: Dense 층과 마찬가지로 맨 처음 층에는 입력되는 값을 알려주어야 합니다. 
# (행, 열, 색상 또는 흑백) 형식으로 정합니다. 색상이면 3, 흑백이면 1을 지정
# 네 번째 인자: 사용할 활성화 함수를 정의합니다.
model.add(Conv2D(64, (3, 3), activation='relu'))
# 커널의 수 는 64개, 커널의 크기는 3 X 3, 활성화 함수는 렐루를 사용하는 컨볼루션 층을 추가
model.add(MaxPooling2D(pool_size=(2,2)))
# pool_size를 통해 풀링 창의 크기는 가로 2, 세로 2 
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 모델 실행 환경을 설정합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 모델 최적화를 위한 설정 구간입니다.
modelpath = "./MNIST_CNN.hdf5"
# val_loss의 값을 모니터링하고 verbose=1로 저장되면 메시지 출력하고, save_best_only true 값으로 모니터 기준 값에서 가장 좋은 모델이 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True) 
# val_loss의 값을 모니터링 하고 10번 이상 모델 성능이 향상되지 않으면 자동으로 학습 중단합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10) 

# 모델을 실행합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
# validation_split 0.25로 20%를 검증셋을 하고 callbacks 함수를 early_stopping_callback, checkpointer를 사용합니다.

# 테스트 정확도를 출력합니다.
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 검증셋과 학습셋의 오차를 저장합니다.
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프로 표현합니다.
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시합니다.
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
