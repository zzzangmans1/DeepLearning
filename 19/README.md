# 19장 세상에 없는 얼굴 GAN, 오토인코더

**생성적 적대 신경망**(Generative Adversarial Networks), 중려서 '**GAN**(간)'이라고 부르는 알고리즘을 이용해 만든 것들이지요.
GAN은 딥러닝의 원리를 활용해 가상 이미지를 생성하는 알고리즘입니다.
GAN이라는 이름에는 적대적(adversairal, 서로 대립 관계에 있는)이란 단어가 들어 있는데, 이것은 GAN 알고리즘의 성격을 잘 말해 줍니다.
진짜 같은 가짜를 만들기 위해 GAN 알고리즘 내부에서는 '적대적' 경합을 진행하기 때문입니다.
이 적대적 경합을 쉽게 설명하기 위해 GAN의 아이디어를 처음으로 제안한 이안 굿펠로는 그의 논문에서 위조지폐범과 경찰의 예를 들었습니다.
진짜 지폐와 똑같은 위조지폐를 만들기 위해 애쓰는 위조지폐범과 이를 가려내기 위한 노력하는 경찰 사이의 경합이 결국 더 정교한 위조지폐를 만들어 낸다는 것이지요.
한쪽은 가짜를 만들고, 한쪽은 진짜와 비교하는 경합 과정을 이용하는 것이 바로 GAN의 원리입니다.

가짜를 만들어 내는 파트를 '생성자(Generator)', 진위를 가려내는 파트를 '판별자('Discriminator)'라고 합니다.
이러한 기본 구조 위에 여러 아이디어를 더한 GAN의 변형 알고리즘들이 지금도 계속해서 발표되고 있습니다.
페이스북의 AI 연구 팀이 만들어 발표한 DCGAN(Deep Convolutional GAN)도 그중 하나입니다.
DCGAN은 우리가 앞서 배운 컨볼루션 신경망(CNN)을 GAN에 적용한 알고리즘인데, 지금의 GAN이 되게끔 해주었다고 해도 과언이 아닐 만큼 불안정하던 초기의 GAN을 크게 보완해 주었습니다.

## 1 가짜 제조 공장, 생성자

**생성자**(generator)는 가상의 이미지를 만들어 내는 공장입니다.
처음에는 랜덤한 픽셀 값으로 채워진 가짜 이미지로 시작해서 판별자의 판별 결과에 따라 지속적으로 업데이트하며 점차 원하는 이미지를 만들어 갑니다.
DCGAN은 생성자가 가짜 이미지를 만들 때 컨볼루션 신경망(CNN)을 이용한다고 했습니다.
우리는 컨볼루션 신경망을 이미 배웠는데 DCGAN에서 사용되는 컨볼루션 신경망은 앞서 나온 것과 조금 차이가 있습니다.
먼저 옵티마이저를 사용하는 최적화과정이나 컴파일하는 과정이 없다는 것입니다.
판별과 학습이 이곳 새엇ㅇ자에서 일어나는 것이 아니기 때문입니다.
이는 이 장에서 차차 다루게 될 것입니다.
또한, 일부 매개변수를 삭제하는 폴링 과정이 없는 대신 앞 장에서 배운 패딩 과정이 포함됩니다.
빈 곳을 채워서 같은 크기로 맞추는 패딩 과정이 여기서 다시 등장하는 이유는 입력 크기와 출력 크기를 똑같이 맞추기 위해서입니다.
커널을 이동하며 컨볼루션 층을 만들 때 이미지의 크기가 처음보다 줄어든다는 것을 떠올려 보면 패딩 과정이 왜 필요한지 알 수 있습니다.
케라스의 패딩 함수는 이러한 문제를 쉽게 처리할 수 있도록 도와줍니다.
padding='same'이라는 설정을 통해 입력과 출력의 크기가 다를 경우 자동으로 크기를 확장해 주고, 확장된 공간에 0을 채워 넣을 수 있습니다.

패딩 외에도 알아야 할 것들이 몇 가지 더 있습니다.
DCGAN의 제안자들은 학습에 꼭 필요한 옵션들을 제시했는데, 그중 하나가 **배치 정규화**(Batch Normalization)라는 과정입니다.
배치 정규화란 입력 데이터의 평균이 0, 분산이 1이 되도록 재배치하는 것인데, 다음 층으로 입력될 값을 일정하게 재배치하는 역할을 합니다.
이 과정을 통해 층의 개수가 늘어나도 안정적인 학습을 진행할 수 있습니다.
케라스는 이를 쉽게 적용할 수 있게끔 BatchNormalization() 함수를 제공합니다.
또한, 생성자의 활성화 함수로는 ReLU() 함수를 쓰고 판별자로 넘겨주기 직전에는 tanh() 함수를 쓰고 있습니다.
tanh() 함수를 쓰면 출력되는 값을 -1에서1 사이로 맞출 수 있습니다.
판별자에 입력될 MNIST 손글씨의 픽셀 범위도 -1에서1로 맞추면 판별 조건이 모두 갖추어집니다.
지금까지 설명한 내용을 코드로 정리하면 다음과 같습니다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation

generator = Sequential() # 모델 이름을 generator로 정하고 Sequential() 함수를 호출
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2))) # 1
generator.add(BatchNormalization()) # 2
generator.add(Reshape((7, 7, 128))) # 3
generator.add(UpSampling2D()) # 4
generator.add(Conv2D(64, kernel)size=5, padding='same')) # 5
generator.add(BatchNormalization()) # 6
generator.add(Activation(LeakyReLU(0.2))) # 7
generator.add(UpSampling2D()) # 8
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh')) # 9
```

먼저 **1**부터 차례로 확인해 보겠습니다.

여기서 128은 임의로 정한 노드의 수입니다.
128이 아니어도 충분한 노드를 마련해 주면 됩니다.
input_dim=100은 100차원 크기의 랜덤 벡터를 준비해 집어넣으라는 의미입니다.
꼭 100이 아니어도 좋습니다.
여기서 주의할 부분은 7X7입니다.
이는 이미지의 최초 크기를 의미합니다.
MNIST 손글씨 이미지의 크기는 28 X 28인데, 왜 7 X 7 크기의 이미지를 넣어 줄까요?
**4**와 **8**을 보면 답이 있습니다.
UpSampling2D() 함수를 사용했습니다.
UpSampling2D() 함수는 이미지의 가로세로 크기를 두 배씩 늘려 줍니다.
7 X 7 **4** 레이어를 지나며 그 크기가 14 X 14가 되고, **8** 레이어를 지나며 28 X 28이 되는 것입니다.
이렇게 작은 크기의 이미지를 점점 늘려 가면서 컨볼루션 층(**5**, **9**)을 지나치게 하는 것이 DCGAN의 특징입니다.
**3**은 컨볼루션 레이어가 받아들일 수 있는 형태로 바꾸어 주는 코드입니다. 
Conv2D() 함수의 input_shape 부분에 들어갈 형태로 정해줍니다.
**4**, **5** 그리고 **8**, **9**는 두 배씩 업샘플링을 한 후 컨볼루션 과정을 처리합니다.
커널 크기로 5로 해서 5 X 5 크기의 커널을 썼습니다.
바로 앞서 설명했듯이 padding='same' 조건때문에 모자라는 부분은 자동으로 0이 채워집니다.
**1**과 **7**에서 활성화 함수로 LeakyReLU를 썼습니다.
GAN에서는 기존에 사용하던 ReLU() 함수를 쓸 경우 학습이 불안정해지는 경우가 많아, ReLU()를 조금 변형한 LeakyReLU() 함수를 씁니다.

0.2로 설정하면 0보다 작을 경우 0.2를 곱하라는 의미입니다.

**2**, **6**에서는 데이터의 배치를 정규 분포로 만드는 배치 정규화가 진행됩니다.
끝으로 **9**에서 한 번 더 컨볼루션 과정을 거친 후 판별자로 값을 넘길 준비를 마칩니다.
앞서 이야기한 대로 활성화 함수는 tanh() 함수를 썼습니다.

## 2 진위를 가려내는 장치, 판별자

이제 생성자에서 넘어온 이미지가 가짜인지 진짜인지를 판별해 주는 장치인 **판별자**(discriminator)를 만들 차례입니다.
이 부분은 컨볼루션 신경망의 구조를 그대로 가지고 와서 만들면 됩니다.
컨볼루션 신경망이란 원래 무언가를(예를 들어 개와 고양이 사진을)구별하는 데 최적화된 알고리즘이기 때문에 그 목적 그대로 사용하면 되는 것이지요.
진짜(1) 아니면 가짜(0), 둘 중 하나를 결정하는 문제이므로 컴파일 부분은 14장에서 사용된 이진 로스 함수(binary_crossentropy)와 최적화 함수(adam)를 그대로 쓰겠습니다.
16장에서 배웠던 드롭아웃(Dropout(0.3))도 다시 사용하고, 앞 절에서 다룬 배치 정규화와 패딩도 그대로 넣어 줍니다.
주의할 점은 이 판별자는 가짜인지 진짜인지를 판별만 해 줄 뿐, 자기 자신이 학습을 해서는 안 된다는 것입니다.
판별자가 얻은 가중치는 판별자 자신이 학습하는 데 쓰이는 것이 아니라 생성자로 넘겨주어 생성자가 업데이트된 이미지를 만들도록 해야 합니다.
따라서 판별자를 만들 때는 가중치를 저장하는 학습 기능을 꺼 주어야 합니다.
모든 과정을 코드로 정래히 보면 다음과 같습니다.
```python
from tensorflow.keras.models import Sequential()
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Flatten, Dense

# 모델 이름을 discriminator로 정하고 Sequential() 함수를 호출합니다.
discriminator = Sequential() 
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28, 28, 1), padding="same")) # 1
discriminator.add(Activation(LeakyReLU(0.2))) # 2
discriminator.add(Dropout(0.3)) # 3
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same")) # 4
discriminator.add(Activation(LeakyReLU(0.2))) # 5
discriminator.add(Dropout(0.3)) # 6
discriminator.add(Flatten()) # 7
discriminator.add(Dense(1, activation='sigmoid')) # 8
discriminator.compile(loss='binary_crossentropy', optimizer='adam') # 9
discriminator.trainable = False # 10
```

먼저 1, 4를 살펴보면 노드의 수는 각각 64개, 128개로 정했고, 커널 크기는 5로 설정해 5 X 5 커널이 사용된다는 것을 알 수 있습니다.

여기에 strides 옵션이 처음 등장했습니다. 
strides는 커널 윈도를 몇 칸씩 이동시킬지 정하는 옵션입니다.
특별한 설정이 없으면 커널 윈도는 한 칸씩 움직입니다.
strides=2라고 설정했다는 것은 커널 윈도를 두 칸씩 움직이라는 뜻입니다.
strides를 써서 커널 윈도를 여러 칸 움직이게 하는 이유는 무엇일까요? 가로세로 크기가 더 줄어들어 새로운 특징을 뽑아 주는 효과가 생기기 때문입니다.
드롭아웃이나 폴링처럼 새로운 필터를 적용한 효과가 생기는 것입니다.
생성자에서는 출력 수를 28로 맞추어야 하기 때문에 오히려 업샘플링을 통해 가로세로의 수를 늘려 주었지만 판별자는 진짜와 가짜만 구분하면 되기 때문에 그럴 필요가 없습니다.
strides나 드롭아웃(3, 6) 등 차원을 줄여 주는 기능을 적극적으로 사용하면서 컨볼루션 신경망 본래의 목적을 달성하면 됩니다.
2, 5는 활성화 함수로 LeakyReLU() 함수를 사용한 것을 보여 줍니다.
7, 8은 가로 X 세로의 2차원으로 진행된 과정을 1차원으로 바꾸어 주는 Flatten() 함수와 마지막 활성화 함수로 sigmoid() 함수를 사용하는 과정입니다.
판별의 결과가 진짜(1) 혹은 가짜(0), 둘 중에 하나가 되어야 하므로 sigmoid() 함수를 썼습니다.
9에서는 이제 이진 로스 함수(binary_crossentropy)와 최적화 함수(adam)를 써서 판별자에 필요한 준비를 마무리합니다.
10에서는 앞서 설명한 대로 판별이 끝나고 나면 판별자 자신이 학습되지 않게끔 학습 기능을 꺼 줍니다.
discriminator.trainable = False 라는 옵션으로 이를 설정할 수 잇습니다.

## 3 적대적 신경망 실행하기

이제 생성자와 판별자를 연결시키고 학습을 진행하며 기타 여러 가지 옵션을 설정하는 순서입니다.
생성자와 판별자를 연결시킨다는 것은 생성자에서 나온 출력을 판별자에 넣어서 진위 여부를 판별하게 만든다는 의미입니다.

생성자 G()에 입력 값 input을 넣은 결과는 G(input)입니다.
이것을 판별자 D()에 넣은 결과는 D(G(input))이 됩니다.
생성자는 D(G(input))이 참(1)이라고 주장하지만, 판별자는 실제 데이터인 x로 만든 D(x)만이 참이라고 여깁니다.
그러다 학습이 진행될수록 생성자가 만든 G(input)이 실제와 너무 가까워져서 이것으로 만든 D(G(input))과 실제 데이터로 만든 D(x)를 잘 구별하지 못하게 되어 정확도가 0.5에 가까워질 때, 비로소 생성자는 자신의 역할을 다하게 되어 학습은 종료됩니다.
이제 이것을 코드로 만들겠습니다.

```python
ginput = Input(shape=100,)) # 1
dis_output = discriminator(generator(ginput)) # 2
gan = Model(ginput, dis_output) # 3
gan.compile(loss='binary_crossentropy', optimizer='adam') # 4
```

1은 랜덤한 100개의 벡터를 케라스 Input() 함수에 집어넣어 생성자에 입력할 ginput을 만드는 과정입니다.
2는 생성자 모델 generator()에 1에서 만든 ginput을 입력합니다.
그 결과 출력되는 28X28 크기의 이미지가 그대로 판별자 모델 discriminator()의 입력 값으로 들어갑니다.
판별자는 이 입력 값을 가지고 참과 거짓을 판별하는데, 그 결과를 dis_output이라고 하겠습니다.
3에서는 케라스의 Model() 함수를 이용해 ginput 값과 2에서 구한 dis_output 값을 넣어 gan이라는 이름의 새로운 모델을 만듭니다.
4에서는 참과 거짓을 구분하는 이진 로스 함수(binary_crossentropy)와 최적화 함수(adam)를 사용해 3에서 만든 gan 모델을 컴파일합니다.
드디어 생성자와 판별자를 연결하는 gan 모델까지 만들었습니다.
이제 지금까지 모든 과정을 실행할 함수를 만들 차례입니다.
gan_train() 함수를 사용해 학습이 진행되도록 하겠습니다. 이때 변수는 epoch, batch_size 그리고 중간 과정을 저장할 때 몇 번마다 한 번씩 저장할지 정하는 saving_interval 이렇게 세 가지로 정합니다.
판별자에서 사용할 MNIST 손글씨 데이터도 불러 줍니다.
앞서 생성자 편에서 tanh() 함수를 사용한 이유는 지금 불러올 이 데이터의 픽셀 값 -1에서1 사이의 값으로 지정하기 위해서였습니다.
0에서255의 값으로 되어 있는 픽셀 값을 -1에서1 사이의 값으로 바꾸려면 현재의 픽셀 값에서 127.5를 뺀 후 127.5로 나누면 됩니다.

```python
# 실행 함수를 선언합니다.
def gan_train(epoch, batch_size, saving_interval): # 세 가지 변수 지정

# MNIST 데이터 불러오기
  # MNIST 데이터를 다시 불러와 이용합니다. 단, 테스트 과정은 필요 없고
  # 이미지만 사용할 것이기 때문에 X_train만 호출합니다.
  (X_train, _), (_, _) = mnist.load_data()
  
  # 가로 28픽셀, 세로 28픽셀이고 흑백이므로 1을 설정합니다.
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
  
  # 0~255 사이 픽셀 값에서 127.5를 뺀 후 127.5로 나누면 -1~1 사이 값으로 바뀝니다.
  X_train = (X_train - 127.5) / 127.5
```

batch_size는 한 번에 몇 개의 실제 이미지와 몇 개의 가상 이미지를 판별자에 넣을지 결정하는 변수입니다.
먼저 batch_size만큼 MNIST 손글씨 이미지를 랜덤하게 불러와 판별자에 집어넣는 과정은 다음과 같습니다.
실제 이미지를 입력했으므로 '모두 참(1)'이라는 레이블을 붙입니다.

```python
import numpy as np

true = np.ones((batch_size, 1)) # 1
idx = np.random.randint(0, X_train.shape[0], batch_size) # 2
imgs = X_train[idx] # 3
d_loss_real = discriminator.train_on_batch(imgs, true) # 4
```

1에서는 '모두 참(1)'이라는 레이블 값을 가진 배열을 만듭니다. batch_size 길이만큼 만들어 4에서 사용합니다.
2에서는 넘파이 라이브러리의 random() 함수를 사용해서 실제 이미지를 랜덤하게 선택해 불러옵니다.
np.random.randint(a, b, c)는 a부터 b까지 숫자 중 하나를 랜덤하게 선택해 가져오는 과정을 c번 반복하라는 의미입니다.
0부터 X_train 개수 사이의 숫자를 랜덤하게 선택해 batch_size만큼 반복해서 가져오게 했습니다.
3에서는 2에서 선택된 숫자에 해당하는 이미지를 불러옵니다.
4에서는 판별자 모델에 train_on_batch() 함수를 써서 판별을 시작합니다.
train_on_batch(x, y) 함수는 입력 값(x)과 레이블(y)을 받아서 딱 한 번 학습을 실시해 모델을 업데이트합니다.
3에서 만든 이미지를 x에 넣고 1에서 만든 배열을 y에 놓아 준비를 마칩니다.
실제 이미지에 이어서 이번에는 생성자에서 만든 가상의 이미지를 판별자에 넣겠습니다.
가상의 이미지는 '모두 거짓(0)'이라는 레이블을 준비해 붙입니다.
학습이 반복될수록 가짜라는 레이블을 붙인 이미지들에 대한 예측 결과가 거짓으로 나올 것입니다.

``` python
fake = np.zeros((batch_size, 1)) # 1 
noise = np.random.normal(0, 1, (batch_size, 100)) # 2
gen_imgs = generator.predict(noise) # 3
d_loss_fake = discriminator.train_on_batch(gen_imgs, fake) # 4
```

1에서는 '모두 거짓(0)'이라는 레이블 값을 가진 열을 batch_size 길이만큼 만듭니다.
2에서는 생성자에 집어넣을 가상 이미지를 만듭니다.
정수가 아니기 때문에 np.random.normal() 함수를 사용했습니다.
조금 전과 마찬가지로 np.random.normal(a, b, c) 형태를 가지며 a부터 b까지 실수 중 c개를 랜덤으로 뽑으라는 의미입니다.
여기서 c 자리에 있는 (batch_size, 100)은 batch_size만큼 100열을 뽑으라는 의미입니다.
3에서는 2에서 만들어진 값이 생성자에 들어가고 결괏값이 gen_imgs로 저장됩니다.
4에서는 3에서 만든 값에 1에서 만든 '모두 거짓(0)'이라는 레이블이 붙습니다.
이대로 판별자로 입력됩니다.
이제 실제 이미지를 넣은 d_loss_real과 가상 이미지를 입력한 d_loss_fake가 판별자에서 번갈아 가며 진위를 판단하기 시작합니다. 
각각 게산되는 오차의 평균을 구하면 판별자의 오차 d_loss는 다음과 같이 정리됩니다.

``` python
# d_loss_real, d_loss_fake 값을 더해 둘로 나눈 평균이 바로 판별자의 오차
d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
```

이제 마지막 단계입니다. 판별자와 생성자를 연결해서 만든 gan 모델을 이용해 생성자의 오차, g_loss를 구하면 다음과 같습니다.
역시 train_on_batch() 함수와 앞서 만든 gen_imgs를 사용합니다.
생성자의 레이블은 무조건 참(1)이라 해놓고 판별자로 넘깁니다.
따라서 이번에도 앞서 만든 true 배열로 레이블을 붙입니다.

``` python
g_loss = gan.train_on_batch(noise, true)
```

그리고 학습이 진행되는 동안 생성자와 판별자의 오차가 출력되게 하겠습니다.

``` python
print('epoch:%d' % i, 'd_loss:%.4f' % d_loss, 'g_loss:%.4f' % g_loss)
```

이제 실행할 준비를 마쳤습니다.
앞서 배운 GAN의 모든 과정을 한곳에 모으면 다음과 같습니다.

[실습1 GAN 모델 만들기](https://github.com/zzzangmans1/DeepLearning/blob/main/19/19_1.py)

![image](https://user-images.githubusercontent.com/52357235/179204200-0301627a-3b93-41af-95ce-402cdf3e6c89.png)

![image](https://user-images.githubusercontent.com/52357235/179204152-f0b4fbad-0399-4b85-930e-a32698263091.png)

## 4 이미지의 특징을 추출하는 오토인코더

딥러닝을 이용해 가상의 이미지를 만드는 또 하나의 유명한 알고리즘이 있습니다.
바로 **오토인코더**(Auto-Encoder, AE)입니다.
지금까지 설명한 GAN을 이해했다면 오토인코더의 핵심적인 부분은 이미 거의 이해한 셈입니다.

오토인코더는 GAN과 비슷한 결과를 만들지만, 다른 성질을 지니고 있습니다.
GAN이 세상에 존재하지 않는 완전한 가상의 것을 만들어 내는 반면에, 오토인코더는 입력 데이터의 특징을 효율적으로 담아낸 이미지를 만들어 냅니다.
예를 들어 GAN으로 사람의 얼굴을 만들면 진짜 같아 보여도 실제로는 존재하지 않는 완전한 가상 이미지가 만들어집니다.
하지만 오토인코더로 사람의 얼굴을 만들 경우 초점이 좀 흐릿하고 윤곽이 불명확하지만 사람의 특징을 유추할 수 있는 것들이 모여 이미지가 만들어집니다.

그렇다면 오토인코더는 과연 어디에 활용할 수 있을까요? 영상 의학 분야 등 아직 데이터 수가 충분하지 않은 분야에서 사용될 수 있습니다.
학습 데이터는 현실 세계의 정보를 담고 있어야 하므로,  세상에 존재하지 않는 가상의 것을 집어넣으면 예상치 못한 결과를 가져올 수 있습니다.
하지만 데이터의 특징을 잘 담아내는 오토인코더라면 다릅니다.
부족한 학습 데이터 수를 효과적으로 늘려 주는 효과를 기대할 수 있지요.
오토인코더의 학습은 GAN의 학습보다 훨씬 쉽습니다. 
이전 절에서 GAN의 원리를 이해했다면 매우 수월하게 익힐 수 있을 것입니다.

입력한 이미지와 똑같은 크기로 출력층을 만들었습니다.
그리고 입력층보다 적은 수의 노드를 가진 은닉층을 중간에 넣어서 차원을 줄여 줍니다.
이때 소실된 데이터를 복원하기 위해 학습을 시작하고, 이 과정을 통해 입력 데이터의 특징을 효율적으로 응축한 새로운 출력이 나오는 원리입니다.
가상 핵심이 되는 인코딩과 디코딩 과정을 코딩해 보면 다음과 같습니다.

```python
# 생성마 모델 만들기
autoencoder = Sequential()

# 인코딩 부분
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu')) # 1
autoencoder.add(MaxPooling2D(pool_size=2, padding='same')) # 2
autoencoder.add(Conv2D(8, kernel_size=3, activation='relu', padding='same')) # 3
autoencoder.add(MaxPooling2D(pool_size=2, padding='same') # 4
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu')) # 5

# 디코딩 부분
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu')) # 6
autoencoder.add(UpSampling2D()) # 7
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu')) # 8
autoencoder.add(UpSampling2D()) # 9
autoencoder.add(Conv2D(16, kernel_size=3, activation='relu')) # 10
autoencoder.add(UpSampling2D()) # 11
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')) # 12

# 전체 구조 확인
autoencoder.summary() # 13
```

1에서5는 입력된 값의 차원을 축소시키는 인코딩 부분이고 6에서12는 다시 차원을 점차 늘려 입력 값과 똑같은 크기의 출력 값을 내보내는 디코딩 부분입니다.
두 부분이 하나의 Sequential() 함수로 쭉 이어져 오토인코더 모델을 만듭니다.
인코딩 파트에서 입력 크기를 줄이는 방법으로 맥스 풀링을 사용했습니다(2, 4).
반대로 디코딩 부분에서는 크기를 늘리기 위해 앞에서 배운 UpSampling을 썼습니다(7, 9, 11).

여기서 놓치지 말아야 할 것은 1에서 입력된 28X28 크기가 층을 지나면서 어떻게 바뀌는지 파악하는 것입니다.
입력된 값을 MaxPooling 층 2, 4를 지나면서 절반씩 줄어들 것이고, Upsampling 층 7, 9, 11을 지나면서 두 배로 늘어납니다.
그렇다면 이상한 점이 하나있습니다.
어째서 MaxPooling 층은 두 번이 나오고 Upsampling 층은 세 번이나 나올까요? 이대로라면 처음 입력한 28X28보다 더 크게 출력되는 것은 아닐까요?
해답은 10에 있습니다. 잘 보면 padding 옵션이 없습니다.
크기를 유지시켜 주는 패딩 과정이 없으므로 커널이 적용되면서 크기가 줄어듭니다.
이를 다시 확인하기 위해 전체 구조를 확인해 보면(13) 다음과 같습니다.

![image](https://user-images.githubusercontent.com/52357235/179203425-d8eeda9a-4b1e-4ac7-bd94-ecb8f06448c7.png)

전체 구조에서 14에서 15로 넘어갈 때 다른 Conv2D 층과 달리 벡터 값이 줄어들었음에 주의해야 합니다.
15의 Conv2D 층에는 padding이 적용되지 않았고 kernel_size = 3이 설정되어 있으므로 3 X 3 커널이 훑고 지나가면서 벡터의 차원을 2만큼 줄였습니다.
마지막 층의 벡터 값이 처음 입력 값과 같은 28 X 28 크기가 되는 것을 확인하면 모든 준비가 된 것입니다.

[실습1 오토인코더 실습하기]()


