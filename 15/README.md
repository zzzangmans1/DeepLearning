 #15장 실제 데이터로 만들어 보는 모델

지금까지 한 실습은 참 또는 거짓을 맞히거나 여러 개의 보기 중 하나를 예측하는 분류 문제였습니다.
그런데 이번에는 수치를 예측하는 문제입니다. 준비된 데이터는 아이오와주 에임스 지역에서 2006년부터 2010년까지 거래된 실제 부동산 판매 기록입니다. 주거 유형, 차고, 자재 및 환경에 관한 80개의 서로 다른 속성을 이용해 집의 가격을 예측해 볼 예정인데 오랜 시간 사람이 일일이 기록하다 보니 빠진 부분도 많고, 집에 따라 어떤 항목은 범위에서 너무 벗어나 있기도 하며, 또 가격과는 관계가 없는 정보가 포함되어 있기도 합니다. 
현장에서 만나게 되는 이런 류의 데이터를 어떻게 다루어야 하는지 이 장에서 학습해 보겠습니다.

##1  데이터 파악하기

- import pandas as pd
- 필요한 api 를 불러옵니다.
- !git clone https://github.com/taehojo/data.git
- 깃허브에 준비된 데이터를 가져옵니다.
- df = pd.read_csv("./data/house_train.csv")
- 집 값 데이터를 불러옵니다.
- df
- 출력합니다.

---
