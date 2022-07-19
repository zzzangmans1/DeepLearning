from tensorflow.keras.models import load_model

import pandas as pd
import time

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git
  
# 캐글에서 내려받은 테스트셋을 불러옵니다.
kaggle_test = pd.read_csv("./data/house_test.csv")

# 카테고리형 변수를 0과 1로 이루어진 변수로 바꿉니다.
kaggle_test = pd.get_dummies(kaggle_test)

# 결측치를 전체 칼럼의 평균으로 대체해 채워 줍니다.
kaggle_test = kaggle_test.fillna(kaggle_test.mean())

# 집 값을 제외한 나머지 열을 저장합니다.
cols_kaggle = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
K_test = kaggle_test[cols_kaggle]

# 앞서 15장에서 만든 모델을 불러옵니다.
model = load_model("./data/model/Ch15-house.hdf5")

# ID와 예측 값이 들어갈 빈 리스트를 만듭니다.
ids = []

# 불러온 모델에 K_test를 적용하고 예측 값을 만듭니다.
Y_prediction = model.predict(K_test).flatten()
for i in range(len(K_test)):
  id = kaggle_test['Id'][i]
  prediction = Y_prediction[i]
  ids.append([id, prediction])
  
# 테스트 결과의 저장 환경을 설정합니다.
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = str(timestr) # 파일명을 연월일-시분초로 정합니다.
outdir = './' # 파일이 저장될 위치를 지정합니다.

# Id와 집 값을 csv 파일로 저장합니다.
df = pd.DataFrame(ids, columns=["Id", "SalePrice"])
df.to_csv(str(outdir + filename + '_submission.csv'), index=False)

