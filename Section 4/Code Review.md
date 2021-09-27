# Code Review

## MMDetection 튜토리얼
```!pip install mmcv-full``` 코드로 MMDetection 모듈을 다운로드 받을 수 있음.
- 하지만, 약 10여분이 소요되어 Colab으로 실행할 시, 매번 다운로드 받아야 함.

```
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install
```
- 상기 코드를 통해 mmdetection github에 접속하여 필요한 setup을 다운로드 할 수 있음.
- mmcv 패키지를 모두 다운로드 받은 후에는 Runtime Restart를 통해 변수를 초기화해야 패키지를 사용할 수 있음.


```
# 아래를 수행하기 전에 kernel을 restart 해야 함. 
from mmdet.apis import init_detector, inference_detector
import mmcv
```

Pretrained된 모델을 다운로드 받을 경로를 생성하고, ```!wget``` 명령어로 다운로드 받음.
```
# pretrained weight 모델을 다운로드 받기 위해서 mmdetection/checkpoints 디렉토리를 만듬. 
!cd mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth 
#1x는 epoch를 12번 했다. 2x는 24번 진행함.
```

경로 내 파일 확인
```
!ls -lia /content/mmdetection/checkpoints
```

학습을 위한 Config 설정을 진행하고, 경로를 /content/mmdetection/ 으로 변경하여 모델을 생성해줌.
```
# config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정. 
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# config 파일과 pretrained 모델을 기반으로 Detector 모델을 생성. 
from mmdet.apis import init_detector, inference_detector
#Pretrained 모델을 가져와서 로딩함.
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# mmdetection은 상대 경로를 인자로 주면 무조건 mmdetection 디렉토리를 기준으로 함. 
%cd mmdetection
#<- 기본 경로가 /content 로 되어 있기 때문에 삳대 경로를 지정해주어야 함.

from mmdet.apis import init_detector, inference_detector

# init_detector() 인자로 config와 checkpoint를 입력함. 
model = init_detector(config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
```

Detection할 이미지 로딩
```
%cd /content/

import cv2
import matplotlib.pyplot as plt
img = '/content/mmdetection/demo/demo.jpg'

img_arr  = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.imshow(img_arr)
```
![image](https://user-images.githubusercontent.com/74092405/134910159-dd46e0c2-9c1b-40d0-9266-07d0c85b6272.png)

Detection 진행 후, 결과 표시
```
img = '/content/mmdetection/demo/demo.jpg'
# inference_detector의 인자로 string(file경로), ndarray가 단일 또는 list형태로 입력 될 수 있음. 
results = inference_detector(model, img)

# results는 list형으로 coco class의  0부터 79까지 class_id별로 80개의 array를 가짐. 
# 개별 array들은 각 클래스별로 5개의 값(좌표값과 class별로 confidence)을 가짐. 개별 class별로 여러개의 좌표를 가지면 여러개의 array가 생성됨. 
# 좌표는 좌상단(xmin, ymin), 우하단(xmax, ymax) 기준. 
# 개별 array의 shape는 (Detection된 object들의 수, 5(좌표와 confidence)) 임
'''
results[0] <- class 0번에 대한 2차원 array [[xmin, ymin, xmax, ymax, 1번 object의 class confidence],
                                            [xmin, ymin, xmax, ymax, 2번 object의 class confidence],
                                            [xmin, ymin, xmax, ymax, 3번 object의 class confidence]]
'''
results

from mmdet.apis import show_result_pyplot
# inference 된 결과를 원본 이미지에 적용하여 새로운 image로 생성(bbox 처리된 image)
# Default로 score threshold가 0.3 이상인 Object들만 시각화 적용. show_result_pyplot은 model.show_result()를 호출. 
show_result_pyplot(model, img, results)
```
![image](https://user-images.githubusercontent.com/74092405/134910467-a5fcb13f-c7a3-4e56-b236-d3253a2e92dc.png)

해당 코드를 통해 Config를 확인할 수 있음.
```
#print(model.cfg)
print(model.cfg.pretty_text)
'''
num_classes
dataset_type
'''
```

**단, array를 inference할 시에는 원본 array를 BGR 형태로 입력해야 함. 그렇지 않으면 BGR형태의 결과를 얻게 됨.**
```
import cv2

# RGB가 아닌 BGR로 입력
img_arr = cv2.imread('/content/mmdetection/demo/demo.jpg')
#img_arr = cv2.cvtColor(cv2.imread('/content/mmdetection/demo/demo.jpg'), cv2.COLOR_BGR2RGB)
results = inference_detector(model, img_arr)
#RGB로 넣으면 BGR 결과로 나옴

show_result_pyplot(model, img_arr, results)
```
![image](https://user-images.githubusercontent.com/74092405/134910789-cdd544e3-ac23-4507-ad1b-2008a21a4c6c.png)


## KittiTinyDataset


