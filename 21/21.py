from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16

# XAI 알고리즘을 불러오는 부분입니다.
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

# 이미지를 불러와 보여 주는 데 쓰는 라이브러리를 불러옵니다.
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 깃허브에 준비된 데이터를 가져옵니다.
!git clone https://github.com/taehojo/data.git

# 원본 이미지가 들어갈 리스트를 만듭니다.
images_originals = []

# 원본 이미지가 저장된 폴더에서 하나씩 불러와 리스트에 넣습니다.
for img_path in glob.glob('./data/img/*_0.jpg'):
  images_originals.append(mpimg.imread(img_path))
  
# 코랩에서 보여 줄 이미지의 크기
plt.figure(figsize=(20, 20))

# 원본 이미지를 코랩에서 보이게 하기
for i, image_o in enumerate(images_originals):
  plt.subplot(5, 5, i+1)
  plt.imshow(image_o)

# 사전에 학습된 딥러닝 모델 불러오기
model = VGG16(weights="imagenet", include_top=True)

# 원본 이미지 이름과 Imagenet에서의 해당 이미지 인덱스
input_list = ["maltese", "persian_cat", "squirrel_monkey", "grand_piano", "yawl" ]
imagenet_index = ["153", "283", "382", "579", "914"]

# 그레이디언트 CAM 알고리즘 선택
explainer = GradCAM()

# 그레이디언트 CAM 알고리즘이 적용된 이미지가 들어갈 빈 리스트 만드릭
images_cams = []

# 그레이디언트 CAM 알고리즘 실행
for l, i in zip(input_list, imagenet_index):
  # 이미지를 불러오고 내부에서 처리될 이미지의 크기를 설정합니다.
  img = load_img('./data/img/{}_0.jpg'.format(l), target_size=(224, 224))
  img = img_to_array(img) # 이미지를 넘파이 배열로 바꿉니다.
  data = ([img], None)
  # 그레이디언트 CAM이 실행되는 부분입니다.
  grid = explainer.explain(data, model, int(i))
  # 실행 후 저장되는 이름입니다.
  explainer.save(grid, ".", './data/img/{}_cam.jpg'.format(l))

# 그레이디언트 CAM 알고리즘이 적용된 이미지를 불러오는 부분의 시작입니다.
plt.figure(figsize=(20, 20))

for img_path in glob.glob('./data/img/*_cam.jpg'):
  images_cams.append(mpimg.imread(img_path))
  
for i, image_c in enumerate(images_cams):
  plt.subplot(5, 5, i+1)
  plt.imshow(image_c)

# 오클루전 알고리즘 선택
explainer = OcclusionSensitivity()

# 알고리즘이 적용된 이미지가 들어갈 빈 리스트 만들기
images_occ1s = []

# 패치 크기를 정합니다.
patch_size = 40

# 오클루전 알고리즘 실행
for l, i in zip(input_list, imagenet_index):
  img = load_img('./data/img/{}_0.jpg'.format(l), target_size=(224, 224))
  img = img_to_array(img)
  data = ([img], None)
  # 패치 크기 설정이 추가됩니다.
  gid = explainer.explain(data, model, int(i), patch_size)
  explainer.save(grid, ".", './data/img/{}_occ1.jpg'.format(l))
  
# 오클루전 알고리즘이 적용된 이미지를 불러오는 부분의 시작입니다.
plt.figure(figsize=(20, 20))

for img_path in glob.glob('./data/img/*_occ1.jpg'):
  images_occ1s.append(mpimg.imread(img_path))
  
for i, image in enumrate(images_occ1s):
  plt.subplot(5, 5, i+1)
  plt.imshow(image)
