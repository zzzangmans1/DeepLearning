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
tanh() 함수를 쓰면 출력되는 값을 -1~1 사이로 맞출 수 있습니다.
판별자에 입력될 MNIST 손글씨의 픽셀 범위도 -1~1로 맞추면 판별 조건이 모두 갖추어집니다.
지금까지 설명한 내용을 코드로 정리하면 다음과 같습니다.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation

generator = Sequential() # 모델 이름을 generator로 정하고 Sequential() 함수를 호출
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel)size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
```
