# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 추출에 필요한 함수

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 피마 인디언 당뇨병 데이터셋을 불러옵니다.
df = pd.read_csv('./data/pima-indians-diabetes3.csv')

df.head(5) # 맨 윗줄부터 5번째 열까지 출력


df["diabetes"].value_counts() # 칼럼의 값이 몇명씩 있는지 체크하는 함수

df.describe() # 칼럼별로 count : 샘플 수, mean : 평균, std : 표준편차, min : 최솟값, 25% : 25%에 해당하는 값, 50% : 50%에 해당하는 값, 75% : 75%에 해당하는 값, max : 최댓값

df.corr() # 칼럼별로 상관관계 확인하는 함수

# 상관관계를 가시화하는 함수
colormap = plt.cm.gist_heat # 그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12)) # 그래프의 크기를 정합니다.

# heatmap() 함수를 통하여 그래프를 표시
# heatmap() 함수는 두 항목씩 짝을 지은 후 각각 어떤 패턴으로 변화하는지 관찰하는 함수
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True) # vmax : 색상의 밝기 조절 인자, cmap : 미리 정해진 맥플롯립 색상의 설정 값을 불러옵니다.
# 색상 설정 값은 https://matplotlib.org/users/colormaps.html에서 확인할 수 있습니다.
plt.show()

# 그래프를 보고 diabetes 항목과 plasma, bmi 항목이 상관관계가 높다는 것을 알 수 있습니다. 
# 이제 이 두 항목만 따로 떼어 내어 당뇨의 발병 여부와 어떤 관계가 있는지 알아보겠습니다.
# hist() 함수를 이용합니다.
# x 축을 plasma 값을 정상, 당뇨로 나누어 구분해 불러옵니다. 이름은 normal 과 diabetes 로 나눕니다.
# bins : 그래프 개수이다.
# barstacked : 여러 데이터가 쌓여 있는 형태의 막대바를 생성하는 옵션
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], bins=10, histtype='barstacked', label=['normal', 'diabetes']) 
plt.legend()

plt.hist(x=[df.bmi[df.diabetes==0], df.bmi[df.diabetes==1]], bins=10, histtype='barstacked', label=['normal', 'diabetes']) 
plt.legend()
