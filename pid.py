# 필요한 라이브러리를 불러옵니다.
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 피마 인디언의 당뇨병 예측하기 실습

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 피마 인디언 당뇨병 데이터셋을 불러옵니다.
df = pd.read_csv('./data/pima-indians-diabetes3.csv')

X = df.iloc[:,0:8] # 세부 정보를 X로 지정합니다.
y = df.iloc[:,8] # 당뇨병 여부를 y로 지정합니다.

model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))
model.summary() # summary() 함수사용으로 층과 층의 연결을 한눈에 볼 수 있게 해 준다.

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history = model.fit(X, y, epochs=100, batch_size=5)
