## Coco Dataset 활용 yolov3 학습
### Ultralytics Yolo v3 설치 및 사용법
```
!git clone https://github.com/ultralytics/yolov3
!cd yolov3;pip install -qr requirements.txt
```

**/content/yolov3/** 폴더 내의 coco dataset에 맞춘 설정 및 모델 관련 ```*.yaml``` 파일을 불러와 학습에 활용

### wandb (weight and bias) 모듈 설치
- 사용하기 위해서는 Weight and Bias 웹사이트에서 회원 가입을 해야함.
- Colab을 사용하는 계정에서 가입하여 사용하는 것이 좋음.
- 연동 후, 1 2 3 번의 옵션을 통해 그래프를 출력할 지 안 할지를 결정할 수 있음!

### Dataset Config와 Weight 파일의 상대 경로 및 절대 경로
- ```train.py```의 option으로 Dataset config yaml 파일을 지정할 수 있음.
- 파일 명만 입력할 경우, yolov3/data 디렉토리에서 탐색.
- 절대 경로로 입력할 경우, 해당 경로에서 탐색.
- weights option의 경우, 파일명만 입력했을 때는 yolov3 디렉토리에서 탐색하지만, 파일이 없다면 자동으로 yolov3 깃허브에서 다운로드 받음. 절대 경로를 입력한 경우, 해당 경로에 다운로드 됨.
- weights 파일은 yolov3.pt, yolov3-tiny.pt, yolov3-spp.pt (Pretrained)

```
!cd yolov3; python train.py --img 640 --batch 8 --epochs 3 --data coco128.yaml --weights yolov3.pt --nosave --cache
#!cd yolov3; python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights '' --cfg yolov3.yaml --nosave --cache # weight가 공란이면 반드시 cfg 라고 써주어야 함!!
#!cd yolov3; python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov3-tiny.pt --nosave --cache
#!cd yolov3;python train.py --img 640 --batch 16 --epochs 3 --data /content/coco128/coco128.yaml --weights /content/coco128/yolov3-tiny.pt --nosave --cache
#!cd yolov3;python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov3-spp.pt --nosave --cache
```

- dataset이나 labels의 디렉토리명을 변경하고 학습을 진행하려면, 아래와 같이 진행하면 됨.

```
%cd /content
!rm -rf /content/coco128 # coco128 폴더 삭제

# /content/data 디렉토리에 coco128.zip을 download하고 압축 해제
!mkdir /content/data
!wget -O /content/data/coco128.zip https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
!cd /content/data; unzip coco128.zip

!wget -O /content/data/coco128/coco128_renew.yaml https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/coco128_renew.yaml
!cat /content/data/coco128/coco128_renew.yaml

!cd /content/yolov3; python train.py --img 640 --batch 8 --epochs 3 --data /content/data/coco128/coco128_renew.yaml --weights yolov3.pt --nosave --cache


# labels가 있는 폴더는 반드시 폴더명을 labels로 지정해줘야 함!! 
!mv /content/data/coco128/labels /content/data/coco128/labels_chg

!cd /content/yolov3; python train.py --img 640 --batch 8 --epochs 3 --data /content/data/coco128/coco128_renew.yaml --weights yolov3.pt --nosave --cache
```

## Oxford Pet Dataset 활용 yolov3 학습
### Oxford pet 데이터 다운로드
```
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# /content/data 디렉토리를 만들고 해당 디렉토리에 다운로드 받은 압축 파일 풀기.
!mkdir /content/data
!tar -xvf images.tar.gz -C /content/data
!tar -xvf annotations.tar.gz -C /content/data
```

- **Oxford Pet Dataset**은 ObjectName+"번호"로 구성됨
```
# Ultralytics Yolo images와 labels 디렉토리를 train, val 용으로 생성
''' Directory Format이 맞아야 됨!!! -> Ultralytics Annotation format으로 변환 -> Dataset config yaml
[Annotation file 명, center x, y, width, height, (0~1 사이)
...
] '''
!mkdir /content/ox_pet;
!cd /content/ox_pet; mkdir images; mkdir labels;
!cd /content/ox_pet/images; mkdir train; mkdir val
!cd /content/ox_pet/labels; mkdir train; mkdir val

import pandas as pd 

pd.read_csv('/content/data/annotations/trainval.txt', sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
```
![image](https://user-images.githubusercontent.com/74092405/140635921-e3548ebe-7137-479f-8f47-59c7711fbc07.png)
- ```데이터프레임.iterrows()``` 코드를 통해 데이터프레임의 모든 행에 들어있는 정보를 알 수 있음.


#### train/test 데이터 분리 및 class명, 이미지 경로, ann경로 정보를 담은 df를 생성
```
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 전체 image/annotation 파일명을 가지는 리스트 파일명을 입력 받아 메타 파일용 DataFrame 및 학습/검증용 DataFrame 생성. 
def make_train_valid_df(list_filepath, img_dir, anno_dir, test_size=0.1):
    '''class_name(이미지 파일 이름에서 가젿옴), image_path(절대 경로), ann_path(~~~.xml)'''
    pet_df = pd.read_csv(list_filepath, sep=' ', header=None, names=['img_name', 'class_id', 'etc1', 'etc2'])
    #class_name은 image 파일명에서 맨 마지막 '_' 문자열 앞까지에 해당. 
    pet_df['class_name'] = pet_df['img_name'].apply(lambda x:x[:x.rfind('_')])
    
    # image 파일명과 annotation 파일명의 절대경로 컬럼 추가
    pet_df['img_filepath'] = img_dir + pet_df['img_name']+'.jpg'
    pet_df['anno_filepath'] = anno_dir + pet_df['img_name']+'.xml'
    # annotation xml 파일이 없는데, trainval.txt에는 리스트가 있는 경우가 있음. 이들의 경우 pet_df에서 해당 rows를 삭제함. 
    pet_df = remove_no_annos(pet_df)

    # 전체 데이터의 10%를 검증 데이터로, 나머지는 학습 데이터로 분리. 
    train_df, val_df = train_test_split(pet_df, test_size=test_size, stratify=pet_df['class_id'], random_state=2021)
    return pet_df, train_df, val_df

# annotation xml 파일이 없는데, trainval.txt에는 리스트가 있는 경우에 이들을 dataframe에서 삭제하기 위한 함수.
def remove_no_annos(df):
    remove_rows = []
    for index, row in df.iterrows():# 모든 row를 참조
        anno_filepath = row['anno_filepath']
        if not os.path.exists(anno_filepath):
            print('##### index:', index, anno_filepath, '가 존재하지 않아서 Dataframe에서 삭제함')
            #해당 DataFrame index를 remove_rows list에 담음. 
            remove_rows.append(index)
    # DataFrame의 index가 담긴 list를 drop()인자로 입력하여 해당 rows를 삭제
    df = df.drop(remove_rows, axis=0, inplace=False)
    return df


pet_df, train_df, val_df = make_train_valid_df('/content/data/annotations/trainval.txt', 
                                               '/content/data/images/', '/content/data/annotations/xmls/', test_size=0.1)
```

#### Annotation을 Ultralytics yolo format으로 변환
- ```voc xml```파일을 Yolo용 포맷인 ```*.txt```으로 변경하는 함수를 생성
- ```def xml_to_txt(input_xml_file, output_txt_file, object_name)```
- Yolo 학습용 데이터는 정규화가 필요하기 때문에, 이미지의 너비(width)와 높이(height)를 읽어서 정규화 진행. ```convert_yolo_coord```함수 활용
- Yolo 학습용 annotation file을 생성하는 함수 생성 ```make_yolo_anno_file```
- 학습용/검증용 데이터프레임의 경로를 지정한 후, 학습 수행. 

```
# 생성된 Directory 구조에 맞춰 yaml 파일 생성
!wget -O /content/ox_pet/ox_pet.yaml https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/ox_pet.yaml


# Google Drive 접근을 위한 Mount 적용. 
import os, sys 
from google.colab import drive 

drive.mount('/content/gdrive')

# soft link로 Google Drive Directory 연결. 
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생. 
!mkdir "/mydrive/ultra_workdir"


###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. 
!cd /content/yolov3; python train.py --img 640 --batch 8 --epochs 20 --data /content/ox_pet/ox_pet.yaml --weights yolov3.pt --project=/mydrive/ultra_workdir \
                                     --name pet --exist-ok 
'''
--cache 는 이미지를 로딩받아서 메모리를 cache로 넣는데, 사이즈가 크면 감당이 안 될 수 있으므로 작은 데이터가 아니면 사용하지 말 것
--no-save 는 가장 마지막만 저장하게 하는 것인데, 이걸 안 하면 best weight도 같이 저장됨
'''

```

## CVAT를 활용한 annotation file 생성 및 Custom Dataset을 통한 학습
<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140636299-6780aae1-6cc8-4170-a2c9-aeb66114f150.png width = 800></p>

- **CVAT.org** 홈페이지에서 이미지를 업로딩한 후, bounding box를 생성할 수 있음.
- 이를 통해, 생성된 Bbox들을 활용하고자 하는 데이터셋(Pascal VOC, Coco Dataset, Yolov3 등등)에 맞춰 annotation file로 추출할 수 있음.
- 클라우드 기반이다 보니, 이미지 용량에 제한이 있고 속도도 느린 편.
- 이전과 마찬가지로, ultralytics yolov3 모듈을 다운로드 받고, pretrained된 모델을 다운로드 받아 학습을 진행하여 모델을 생성하고, 이를 활용하여 inference를 수행하면 됨.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140636434-48f76d3c-c5a8-42f2-aa22-302bcdd715de.png width = 800></p>


## yolov5를 이용한 BCCD 데이터셋 학습
- Coco 포맷의 데이터를 yolo 포맷으로 변환해줘야 함.
- https://github.com/alexmihalyk23/COCO2YOLO.git 를 약간 수정하여 변환로직 생성 가능.

#### COCO2YOLO 소스코드
```
# https://github.com/alexmihalyk23/COCO2YOLO.git
'''train.json: 'images': {id: ~~ ,
                            w: 640,
                            h: 320,
                            ...,
                            }
                'annotations':{
                    bbox:                    
                },
                image-idx,
                ...,

'''

import json
import os
import shutil

class COCO2YOLO:
  # 소스 이미지 디렉토리와 Json annotation 파일, 타겟 이미지 디렉토리, 타겟 annotation 디렉토리를 생성자로 입력 받음. 
  def __init__(self, src_img_dir, json_file, tgt_img_dir, tgt_anno_dir):
    self.json_file = json_file
    self.src_img_dir = src_img_dir
    self.tgt_img_dir = tgt_img_dir
    self.tgt_anno_dir = tgt_anno_dir
    # json 파일과 타겟 디렉토리가 존재하는지 확인하고, 디렉토리의 경우는 없으면 생성. 
    self._check_file_and_dir(json_file, tgt_img_dir, tgt_anno_dir)
    # json 파일을 메모리로 로딩. 
    self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
    # category id와 이름을 매핑하지만, 실제 class id는 이를 적용하지 않고 별도 적용. 
    self.coco_id_name_map = self._categories()
    self.coco_name_list = list(self.coco_id_name_map.values())
    print("total images", len(self.labels['images']))
    print("total categories", len(self.labels['categories']))
    print("total labels", len(self.labels['annotations']))
  
  # json 파일과 타겟 디렉토리가 존재하는지 확인하고, 디렉토리의 경우는 없으면 생성. 
  def _check_file_and_dir(self, file_path, tgt_img_dir, tgt_anno_dir):
    if not os.path.exists(file_path):
        raise ValueError("file not found")
    if not os.path.exists(tgt_img_dir):
        os.makedirs(tgt_img_dir)
    if not os.path.exists(tgt_anno_dir):
        os.makedirs(tgt_anno_dir)

  # category id와 이름을 매핑하지만, 추후에 class 명만 활용. 
  def _categories(self):
    categories = {}
    for cls in self.labels['categories']:
        categories[cls['id']] = cls['name']
    return categories
  
  # annotation에서 모든 image의 파일명(절대 경로 아님)과 width, height 정보 저장. 
  def _load_images_info(self):
    images_info = {}
    for image in self.labels['images']:
        id = image['id']
        file_name = image['file_name']
        if file_name.find('\\') > -1:
            file_name = file_name[file_name.index('\\')+1:]
        w = image['width']# 정규화를 위해 width와 height를 구함
        h = image['height']
  
        images_info[id] = (file_name, w, h)

    return images_info

  # ms-coco의 bbox annotation은 yolo format으로 변환. 좌상단 x, y좌표, width, height 기반을 정규화된 center x,y 와 width, height로 변환. 
  def _bbox_2_yolo(self, bbox, img_w, img_h):
    # ms-coco는 좌상단 x, y좌표, width, height
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # center x좌표는 좌상단 x좌표에서 width의 절반을 더함. center y좌표는 좌상단 y좌표에서 height의 절반을 더함.  
    centerx = bbox[0] + w / 2
    centery = bbox[1] + h / 2
    # centerx, centery, width, height를 이미지의 width/height로 정규화. 
    dw = 1 / img_w
    dh = 1 / img_h
    centerx *= dw
    w *= dw
    centery *= dh
    h *= dh
    return centerx, centery, w, h
  
  '''
  # image와 annotation 정보를 기반으로 image명과 yolo annotation 정보 가공. 
  Yolo : [centerx, centery, width, height] * 정규화 시킴.
  # 개별 image당 하나의 annotation 정보를 가지도록 변환. '''
  def _convert_anno(self, images_info):
    anno_dict = dict()
    for anno in self.labels['annotations']:
      bbox = anno['bbox']
      image_id = anno['image_id']
      category_id = anno['category_id']

      image_info = images_info.get(image_id)
      image_name = image_info[0]
      img_w = image_info[1]
      img_h = image_info[2]
      yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

      anno_info = (image_name, category_id, yolo_box)
      anno_infos = anno_dict.get(image_id)
      if not anno_infos:
        anno_dict[image_id] = [anno_info]
      else:
        anno_infos.append(anno_info)
        anno_dict[image_id] = anno_infos
    return anno_dict

  # class 명을 파일로 저장하는 로직. 사용하지 않음. 
  def save_classes(self):
    sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
    print('coco names', sorted_classes)
    with open('coco.names', 'w', encoding='utf-8') as f:
      for cls in sorted_classes:
          f.write(cls + '\n')
    f.close()
  # _convert_anno(images_info)로 만들어진 anno 정보를 개별 yolo anno txt 파일로 생성하는 로직. 
  # coco2yolo()에서 anno_dict = self._convert_anno(images_info)로 만들어진 anno_dict를 _save_txt()에 입력하여 파일 생성
  def _save_txt(self, anno_dict):
    # 개별 image별로 소스 image는 타겟이미지 디렉토리로 복사하고, 개별 annotation을 타겟 anno 디렉토리로 생성. 
    for k, v in anno_dict.items():
      # 소스와 타겟 파일의 절대 경로 생성. 
      src_img_filename = os.path.join(self.src_img_dir, v[0][0])
      tgt_anno_filename = os.path.join(self.tgt_anno_dir,v[0][0].split(".")[0] + ".txt")
      #print('source image filename:', src_img_filename, 'target anno filename:', tgt_anno_filename)
      # 이미지 파일의 경우 타겟 디렉토리로 단순 복사. 
      shutil.copy(src_img_filename, self.tgt_img_dir)
      # 타겟 annotation 출력 파일명으로 classid, bbox 좌표를 object 별로 생성. 
      with open(tgt_anno_filename, 'w', encoding='utf-8') as f:
        #print(k, v)
        # 여러개의 object 별로 classid와 bbox 좌표를 생성. 
        for obj in v:
          cat_name = self.coco_id_name_map.get(obj[1])
          # category_id는 class 명에 따라 0부터 순차적으로 부여. 
          category_id = self.coco_name_list.index(cat_name)
          #print('cat_name:', cat_name, 'category_id:', category_id)
          box = ['{:.6f}'.format(x) for x in obj[2]]
          box = ' '.join(box)
          line = str(category_id) + ' ' + box
          f.write(line + '\n')

  # ms-coco를 yolo format으로 변환. 
  def coco2yolo(self):
    print("loading image info...")
    images_info = self._load_images_info()
    print("loading done, total images", len(images_info))

    print("start converting...")
    anno_dict = self._convert_anno(images_info)
    print("converting done, total labels", len(anno_dict))

    print("saving txt file...")
    self._save_txt(anno_dict)
    print("saving done")
```

#### 학습/검증/테스트용 images, labels 디렉토리 생성.
``` 
!mkdir /content/bccd;
!cd /content/bccd; mkdir images; mkdir labels;
!cd /content/bccd/images; mkdir train; mkdir val; mkdir test
!cd /content/bccd/labels; mkdir train; mkdir val; mkdir test
```

#### Dataset용 yaml 파일을 생성하고 학습 수행
- yolo v5는 모델이 yolov5s(small), yolov5m(middle), yolov5l(large), yolov5x(extra large)로 되어있음. weight 인자값으로 이들중 하나를 입력해 줌
```
!wget -O /content/bccd/bccd.yaml https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/bccd.yaml

# Google Drive 접근을 위한 Mount 적용. 
import os, sys 
from google.colab import drive 

drive.mount('/content/gdrive')

# soft link로 Google Drive Directory 연결. 
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생. 
!mkdir "mydrive/TNT Study/2021-02 Study/yolo/ultra_workdir"

###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. large 모델 적용 시 batch size가 8보다 클 경우 colab에서 memory 부족 발생.
### 혈소판의 경우 상대적으로 mAP:0.5~0.95 Detection 성능이 좋지 못함. 백혈구 만큼 학습데이터가 적은것도 이유지만, Object 사이즈가 상대적으로 작음.   
!cd /content/yolov5; python train.py --img 640 --batch 8 --epochs 30 --data /content/bccd/bccd.yaml --weights yolov5l.pt \
                                     --project="mydrive/TNT Study/2021-02 Study/yolo/ultra_workdir" --name bccd --exist-ok

''' 학습 완료 후'''

from collections import Counter

anno_list = train_yolo_converter.labels['annotations']
category_list = [x['category_id'] for x in anno_list]

Counter(category_list)

# Inference 수행

# image 파일 inference 
!cd /content/yolov5;python detect.py --source /content/bccd/images/test/BloodImage_00011.jpg \
                            --weights 'mydrive/TNT Study/2021-02 Study/yolo/ultra_workdir/bccd/weights/best.pt' --conf 0.2 \
                            --project=/content/data/output --name=run_image --exist-ok --line-thickness 2
                            
# test.py가 없어서 val.py로 대신 성능 평가 진행
!cd /content/yolov5; python val.py --weights 'mydrive/TNT Study/2021-02 Study/yolo/ultra_workdir/bccd/weights/best.pt'  --data /content/bccd/bccd.yaml \
                           --project /content/data/output --name=test_result --exist-ok --img 640 --iou 0.65
```

