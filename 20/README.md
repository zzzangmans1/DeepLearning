# 20장 전이 학습을 통해 딥러닝의 성능 극대화하기

딥러닝으로 좋은 성과를 내려면 딥러닝 프로그램을 잘 짜는 것도 중요하지만, 딥러닝에 입력할 데이터를 모으는 것이 더 중요합니다.
기존 머신 러닝과 달리 딥러닝은 스스로 중요한 속성을 뽑아 학습하기 때문에 비교적 많은 양의 데이터가 필요합니다.
하지만 데이터가 충분하지 않은 상황도 발생합니다.
이 장에서는 나만의 프로젝트를 기획하고 실습하는 과정을 따라해 보며, 딥러닝의 데이터양이 충분하지 않을 때 활용할 수 있는 방법들을 배우겠습니다.
여러 방법 중에서 수만 장에 달하는 기존의 이미지에서 학습한 정보를 가져와 내 프로젝트에 활용하는 것을 **전이 학습**(transfer learning)이라고 합니다.
방대한 자료를 통해 미리 학습한 가중치 값을 가져와 내 프로젝트에 사용하는 방법으로 컴퓨터 버전, 자연어 처리 등 다양한 분야에서 전이 학습을 적용해 예측률을 높이고 있습니다.

## 1 소규모 데이터셋으로 만드는 강력한 학습 모델

딥러닝을 이용한 프로젝트는 어떤 데이터를 가지고 있는지, 어떤 목적을 가지고 있는지 잘 살펴보는 것부터 시작합니다.
내가 가진 데이터에 따라 딥러닝 알고리즘을 결정해야 하는데, 딥러닝 및 머신 러닝 알고리즘은 크게 두 가지 유형으로 나뉩니다.
정답을 알려주고 시작하는가 아닌가에 따라 **지도 학습**(supervised learning) 방식과 **비지도 학습**(unsupervised learning) 방식으로 구분되지요.
지금까지 이 책에서 살펴본 폐암 수술 환자의 생존률 예측, 피마 인디언의 당뇨병 예측, CNN을 이용한 MNIST 분류 등은 각 데이터 또는 사진마다 '클래스'라는 정답을 주고 시작했습니다.
따라서 모두 '지도 학습'의 예가 됩니다.
반면 19장에서 배운 GAN이나 오토인코더는 정답을 예측하는 것이 아니라 주어진 데이터의 특성을 찾았기 때문에 '비지도 학습'의 예가 됩니다.
이번에 진행할 프로젝트는 MRI 뇌 사진을 보고 치매 환자의 뇌인지, 일반인의 뇌인지 예측하는 것입니다.
각 사진마다 치매 혹은 일반인으로 클래스가 주어지므로 지도 학습의 예라고 할 수 있겠지요.
이미지를 분류할 것이므로 이미지 분류의 대표적인 알고리즘인 컨볼루션 신경망(CNN)을 선택해 진행하겠습니다.

총 280장으로 이루어진 뇌의 단면 사진입니다.
치매 환자의 특성을 보이는 뇌 사진 140장과 일반인의 뇌 사진 140장으로 구성되어 있습니다.
280개의 이미지 중 160개는 train 폴더에, 120개는 test 폴더에 넣어 두었습니다.
각 폴더 밑에는 ad와 normal이라는 두 개의 폴더가 있는데, 치매 환자의 뇌 사진은 ad 폴더에, 일반인의 뇌 사진은 normal 폴더에 저장했습니다.
앞서 MNIST 손글씨나 로이터 뉴스, 영화 리뷰의 예제들과는 다르게 케라스에서 제공하는 데이터를 불러오는 것이 아니라, 내 데이터를 읽어 오는 것이므로 새로운 함수가 필요합니다.
데이터의 수를 늘리는 ImageDataGenerator() 함수와 폴더에 저장된 데이터를 불러오는 flow_from_directory() 함수를 사용하겠습니다.

ImageDataGenerator() 함수는 주어진 데이터를 이용해 변형된 이미지를 만들어 학습셋에 포함시키는 편리한 기능을 제공합니다.
이미지 데이터의 수를 확장할 때 효과적으로 사용할 수 있습니다.
다음은 함수를 사용한 예입니다.

``` python
train_dategen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=5,
                                   shear_range=0.7,
                                   zoom_range=1.2,
                                   vertical_flip=True,
                                   fill_mode='nearest')
```

-rescale : 주어진 이미지의 크기를 바꾸어 줍니다. 예를 들어 원본 영상이 0~255의 RGB 값을 가지고 있으므로 255로 나누면 0~1의 값으로 변환되어 학습이 좀 더 빠르고 쉬워집니다.
-horizontal_flip, vertical_flip : 주어진 이미지를 수평 또는 수직으로 뒤집습니다.
-zoom_range : 정해진 범위 안에서 축소 또는 확대합니다.
-width_shift_range, height_shift_range : 정해진 범위 안에서 그림을 수평 또는 수직으로 랜덤하게 평행 이동시킵니다.
-rotation_range : 정해진 각도만큼 이미지를 회전시킵니다.
-shear_range : 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환을 합니다.
-fill_mode : 이미지를 축소 또는 회전하거나 이동할 때 생기는 빈 공간을 어떻게 채울지 결정합니다. nearest 옵션을 선택하면 가장 비슷한 색으로 채워집니다.

단, 이 모든 인자를 다 적용하면 불필요한 데이터를 만들게 되어 오히려 학습 시간이 늘어난다는 것에 주의해야 합니다.
주어진 데이터의 특성을 잘 파악한 후 이에 맞게 사용하는 것이 좋습니다.
우리는 좌우의 차이가 그다지 중요하지 않은 뇌 사진을 이용할 것이므로 수평으로 대칭시키는 horizontal_flip 인자를 사용하겠습니다.
그리고 width_shift, height_shift 인자를 이용해 조금씩 좌우로 수평 이동시킨 이미지도 만들어 사용하겠습니다.
참고로 데이터 부풀리기는 학습셋에만 적용하는 것이 좋습니다.
테스트셋은 실제 정보를 그대로 유지하게 하는 편이 과적합의 위험을 줄일 수 있기 때문입니다.
테스트셋은 다음과 같이 정규화만 진행해 줍니다.

``` python
test_datagen = ImageDataGenerator(rescale=1./255)
```

이미지 생성 옵션을 정하고 나면 실제 데이터가 있는 곳을 알려 주고 이미지를 불러오는 작업을 해야 합니다.
이를 위해 flow_from_directory() 함수를 사용하겠습니다.

``` python
train_generator = train_datagen.flow_from_directory(
    './data-ch20/train', # 학습셋이 있는 폭더 위치
    target_size=(150, 150), # 이미지 크기 
    batch_size=5,
    class_mode='binary') # 치매/정상 이진 분류이므로 바이너리 모드로 실행
```

같은 과정을 거쳐서 테스트셋도 생성해 줍니다.

``` python
test_generator = test_datagen.flow_from_directory(
     './data-ch20/test', # 테스트셋이 있는 폴더 위치
     target_size=(150, 150),
     batch_size=5,
     class_mode='binary')
```

모델 실행을 위한 옵션을 만들어 줍니다. 
옵티마이저로 Adam을 선택하는데, 이번에는 케라스 API의 1 optimizers 클래스를 이용해 학습률을 따로 지정해 보았습니다.
조기 중단을 설정하고 model.fit()을 실행하는데, 이때 학습셋과 검증셋을 조금 전 만들어 준 2 train_generator와 3 test_generator로 지정합니다.

``` python
# 모델의 실행 옵션을 설정합니다.
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002), metrics=['accuracy']) # 1

# 학습의 조기 중단을 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# 모델을 실행합니다.
history = model.fit(train_generator, # 2
                    epochs=100,
                    validation_data=test_generator, # 3
                    validation_steps=10,
                    callbacks=[early_stopping_callback]) 
```

** 이 실습에선 사이파이(SciPy) 라이브러리가 필요합니다. 코랩의 경우 기본으로 제공하지만, 주피터 노트북을 이용해 실습 중이라면 다음 명령으로 라이브러리 설치해야 합니다.**
``` python
!pip install Scipy
```
