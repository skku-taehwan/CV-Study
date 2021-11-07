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




