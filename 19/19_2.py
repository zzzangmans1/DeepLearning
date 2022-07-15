from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋을 불러옵니다.
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# 생성자 모델 만들기
autoencoder = Sequential()

# 인코딩 부분
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu')) # 1
autoencoder.add(MaxPooling2D(pool_size=2, padding='same')) # 2
autoencoder.add(Conv2D(8, kernel_size=3, activation='relu', padding='same')) # 3
autoencoder.add(MaxPooling2D(pool_size=2, padding='same')) # 4
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu')) # 5

# 디코딩 부분
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu')) # 6
autoencoder.add(UpSampling2D()) # 7
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu')) # 8
autoencoder.add(UpSampling2D()) # 9
autoencoder.add(Conv2D(16, kernel_size=3, activation='relu')) # 10
autoencoder.add(UpSampling2D()) # 11
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')) # 12

# 컴파일 및 학습을 하는 부분입니다.
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_test, X_test))

# 학습된 결과를 출력하는 부분입니다.
random_test = np.random.randint(X_test.shape[0], size=5)

# 테스트할 이미지를 랜덤으로 호출합니다.
ae_imgs = autoencoder.predict(X_test) # 앞서 만든 오토인코더 모델에 넣습니다.

plt.figure(figsize=(7, 2)) # 출력 이미지의 크기를 정합니다.

for i, image_idx in enumerate(random_test):
  # 랜덤으로 뽑은 이미지를 차례로 나열합니다.
  ax = plt.subplot(2, 7, i+1)
  # 테스트할 이미지를 먼저 그대로 보여 줍니다.
  plt.imshow(X_test[image_idx].reshape(28, 28))
  ax.axis('off')
  ax = plt.subplot(2, 7, 7+i+1)
  # 오토인코딩 결과를 다음 열에 입력합니다.
  plt.imshow(ae_imgs[image_idx].reshape(28, 28))
  ax.axis('off')
plt.show()
