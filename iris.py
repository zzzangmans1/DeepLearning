# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 다중 분류 문제 해결

# 데이터 추출에 필요한 함수

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 피마 인디언 당뇨병 데이터셋을 불러옵니다.
df = pd.read_csv('./data/iris3.csv')

df.head(5) # 맨 윗줄부터 5번째 열까지 출력

#sns.pairplot(df, hue='species'); # pairplot() : 전체 상관도를 볼 수 있는 그래프 출력 
# hue 옵션은 주어진 데이터 중 어떤 카테고리를 중심으로 그래프를 그릴지 정해 주는 옵션
#plt.show() 그래프 출력

X = df.iloc[:,0:4]
y = df.iloc[:,4]

#print(X[0:5]) # X 데이터 5번째 까지 출력
#print(y[0:5]) # y 데이터 5번째 까지 출력

# 먼저 아이리스 꽃의 종류는 세 종류입니다. 그러면 각각의 이름으로 세 개의 열을 만들 후 자신의 이름이 일치하는 경우 1로 나머지는 0으로 바꾸어줍니다.
# 위와 같이 여러 개의 값으로 된 문자열을 0과 1로만 이루어진 형태로 만들어 주는 과정을 원-핫 인코딩이라고 합니다. 
# 원-핫 온키동은 판다스가 제공하는 get_dummies() 함수를 사용하면 간단하게 해낼 수 있습니다.
y = pd.get_dummies(y)

#print(y[0:5])

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax')) # 0과 1의 출력이 아닌 세 가지의 확률을 모두 구해야 하므로 시그모이드 함수가 아닌 softmax 함수가 필요하다. 출력층
model.summary()

# 모델 컴파일 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 손실 함수도 이항이 아니므로 다항 분류에 쓰는 categorical_crossentropy 함수를 써야한다.

history = model.fit(X, y, epochs=50, batch_size=5)
