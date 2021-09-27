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
- 마찬가지로, mmcv-full 패키지를 다운로드 받은 후에 detection을 진행함.
- 다만, mmdetecion의 framework에 집어넣기 위해 annotation 포맷 list 형태로의 변환이 필요함.
- 
![image](https://user-images.githubusercontent.com/74092405/134911098-8572249f-818a-4b80-bd9e-1e07b6a07ee6.png)

- 위의 이미지처럼, 중립 데이터 형태로 변환하여 메모리에 로드해야 함.

```
# 원본 kitti Dataset는 10개의 Class로 되어 있음. 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'
CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
cat2label = {k:i for i, k in enumerate(CLASSES)}
print(cat2label)
cat2label['Car']
```
-> '클래스값: 클래스 이름'의 형태로 나옴.

**CustomDataset에 대한 설정**
```
'''
MMDetection 내부에서 Config와 데이터셋이 어떻게 상호작용하는지 알아야 함!!
-> Config 기반의 패키지이기 때문에 반드시 알아야 함
-> 내부 Framework를 모르면 수정이 어려움
'''
import copy
import os.path as osp
import cv2

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

# 반드시 아래 Decorator 설정 할것.@DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. 
@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):# MMdetection의 CustomDataset을 상속받음
  CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')#클래스명을 꼭 넣어줘야 함 + 반드시 CLASSES = () 구조로 작성해야 함!!
  '''
  def __init__코드가 없음 -> CustomDataset으로부터 상속받겠다는 의미임
  ##### self.data_root: /content/kitti_tiny/ self.ann_file: /content/kitti_tiny/train.txt self.img_prefix: /content/kitti_tiny/training/image_2
  #### ann_file: /content/kitti_tiny/train.txt
  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 이 self.ann_file이 load_annotations()의 인자로 입력
  '''
  def load_annotations(self, ann_file):#ann_file을 받아서 중립데이터 형태로 변환
    print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
    print('#### ann_file:', ann_file)
    cat2label = {k:i for i, k in enumerate(self.CLASSES)}
    image_list = mmcv.list_from_file(self.ann_file)
    # 포맷 중립 데이터를 담을 list 객체 -> 파일을 읽어서 리스트로 만듦.
    data_infos = []
    #img_prefix = '/content/kitti_tiny/training/image_2'
    for image_id in image_list:#for문을 돌면서 포맷 중립 데이터를 담아주는 역할을 함.
      filename = '{0:}/{1:}.jpeg'.format(self.img_prefix, image_id)#이미지를 불러오기 위해서 절대 경로를 사용함.
      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함. 
      image = cv2.imread(filename)
      height, width = image.shape[:2]
      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
      data_info = {'filename': str(image_id) + '.jpeg',
                   'width': width, 'height': height}
      # 개별 annotation이 있는 서브 디렉토리의 prefix 변환. 
      label_prefix = self.img_prefix.replace('image_2', 'label_2')
      # 개별 annotation 파일을 1개 line 씩 읽어서 list 로드 
      lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt'))

      # 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. content는 list의 list형태임.
      # ann 정보는 numpy array로 저장되나 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.   
      content = [line.strip().split(' ') for line in lines]
      # 오브젝트의 클래스명은 bbox_names로 저장. 
      bbox_names = [x[0] for x in content]
      # bbox 좌표를 저장
      bboxes = [ [float(info) for info in x[4:8]] for x in content]

      # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'sms ignore로 별도 저장.
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []

      for bbox_name, bbox in zip(bbox_names, bboxes):
        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        if bbox_name in cat2label:
          gt_bboxes.append(bbox)
          # gt_labels에는 class id를 입력
          gt_labels.append(cat2label[bbox_name])
        else:#Dontcare가 나오면 여기에서 실행됨.
          gt_bboxes_ignore.append(bbox)
          gt_labels_ignore.append(-1)
      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임. 
      data_anno = {
          'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
          'labels': np.array(gt_labels, dtype=np.long),#클래스 id가 들어감, 즉 숫자값으로 나타남.
          'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
          'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
      data_info.update(ann=data_anno)#위에서 만든 data_info를 업데이트해주는 과정.
      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)

    return data_infos
```

Config 설정 및 모델 생성
```
### Config 설정하고 Pretrained 모델 다운로드
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

#check point 파일 다운로드
!cd mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

```

**Config 파일 설정**
```
from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'KittyTinyDataset'#객체 클래스명을 적어주면 됨.
cfg.data_root = '/content/kitti_tiny/'#데이터가 있는 경로 지정
'''
Dataset -> Train, Validation, Test용 데이터로 나눠서 만들어야 함.
'''
# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'KittyTinyDataset'
cfg.data.train.data_root = '/content/kitti_tiny/'#안 써줘도 overwrite하지만 써주는 게 좋음!!
cfg.data.train.ann_file = 'train.txt'#나중에 concate하기 때문에 반드시 1개만 넣어야 함!!!
'''
Annotation은 /content/kitti_tiny/training/label_2 안에 있지만, 파일 하나만 넣을 수 있기 때문에 train.txt로 가져온다.
'''
cfg.data.train.img_prefix = 'training/image_2'

cfg.data.val.type = 'KittyTinyDataset'
cfg.data.val.data_root = '/content/kitti_tiny/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'training/image_2'

cfg.data.test.type = 'KittyTinyDataset'
cfg.data.test.data_root = '/content/kitti_tiny/'
cfg.data.test.ann_file = 'val.txt'
cfg.data.test.img_prefix = 'training/image_2'

# class의 갯수 수정.  80개 -> 4개로 수정
cfg.model.roi_head.bbox_head.num_classes = 4
# pretrained 모델
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
'''
상대 경로로 할 때 유의할 점 
- 데이터의 경우에는 절대 경로로 해주는 것이 좋음!!!
- checkpoints는 상대경로인데, 자연스럽게 mmdetection 아래에 있다고 생각함.
'''

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'
'''
./ 는 content/ 를 의미함.
'''
# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# config 수행 시마다 policy값이 없어지는 bug로 인하여 설정. 
cfg.lr_config.policy = 'step'

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'#mAP가 오래 걸려서 interval을 설정해줘야 함.
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
```


**Dataset 설정 및 학습 진행
```
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]

# 주의, config에 pretrained 모델 지정이 상대 경로로 설정됨 cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# 아래와 같이 %cd mmdetection 지정 필요. 
 
%cd mmdetection 

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# BGR Image 사용 
img = cv2.imread('/content/kitti_tiny/training/image_2/000068.jpeg')

model.cfg = cfg

result = inference_detector(model, img)
show_result_pyplot(model, img, result)
```

**Video Detection 또한 마찬가지 프로세스를 거치면 됨.**
