# Pytorch 기반 주요 Object Detection/Segmentation 패키지

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134906867-2e671f31-ceb0-4753-89b0-424d761b2552.png width = 600></p>


### torch vision
- Code 기반이며, 지원되는 알고리즘이 많지 않음.

### Detectron 2
- Config 기반이며 Facebook에서 만들어짐.

### MM Detection
- Config 기반이며 중국 칭화 대학 중심으로 만들어짐.
- 지원되는 알고리즘이 많으며, 성능이 좋음.
- **효율적인 모듈 설계**, Config 기반으로 데이터 ~ 모델 학습/평가까지 이어지는 파이프라인 적용
- **Config**에 대한 이해가 필수적임.

#### 지원되는 모델

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134907213-5006cab1-d2c4-4c2d-a57e-772bf527d3f8.png width = 600></p>

- Fast RCNN, Faster RCNN, RetinaNet 등의 Prebuilt 모델을 가져와 MMDetection 모델 아키테쳐에 넣어서 학습을 진행할 수 있음.
- **Backbone:** Feature Extractor
- **Neck:** Backbone & Heads를 연결하면서 feature map의 특성을 보다 잘 해석하고 처리할 수 있도록 정제 작업 수행.
- **DenseHead:** F.M.에서 Object의 위치와 분류를 처리.
- **ROIExtractor:** F.M.에서 ROI 정보를 뽑아냄.
- **ROIHead (BBoxHead/MaskHead):** ROI 정보를 기반으로 Object 위치와 분류를 수행함.

**MMDetection의 주요 구성 요소


<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134907632-463db1a5-61f3-478c-a87c-8cffd6c9e3d0.png width = 600></p>

- Config를 지정하는 부분이 모델 구성의 80% 이상을 차지할 정도로 Config에 대한 이해가 필요함.

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134907848-e5670df5-ca66-4b3d-a4a7-b6d8342a37d2.png width = 600></p>

- 학습 도중 Hook(Callback)을 통해 학습에 필요한 여러 설정들을 Customization 할 수 있음.
- 이러한 설정 또한, Configuiration에서 설정.

## Pascal VOC & MS CoCo 데이터의 가장 큰 차이점
- Pascal VOC dataset은 이미지 한 개당 하나의 Annotation 파일이 존재함.
- 하지만, Ms CoCo 데이터셋은 모든 이미지에 대해 단 하나의 Annotation 파일만 존재함.
- -> MMDetection Architecture 또한 단 하나의 Annnotation 파일만을 입력할 수 있어서 사용자 설정 데이터셋을 넣어줄 때, Ms CoCo 데이터셋처럼 하나의 Annotation 파일로 만들어줘야 함.

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134908179-b81a468a-aae5-4a25-b213-21435b0a6387.png width = 600></p>

- 이러한 특징 때문에, Pascal VOC, Ms CoCo, KittitinyDataset 등 다양한 데이터셋을 변환할 수 있는 클래스를 통해 지원함.
- -> CustomDataset 파일을 잘 다룰 수 있어야 함.

### CustomDataset의 구조

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134908476-b9014b8e-04da-4a5c-97b3-57e3a16d3fe0.png width = 600></p>

<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134908759-40fc6f8b-75b3-4738-b518-57a84e4c1a90.png width = 600></p>

- Config 파일에서 꼭 ```ann_file```, ```data_root```, ```img_prefix```를 설정해줘야 함.
- 왜냐하면, MMDetection Framework에서 해당 변수들을 호출하기 때문.

- MMDetection Framework는 CustomDataset 객체를 가져와서 Config에 설정된 주요 값으로 CustomDataset의 객체를 생성함.


<p align="center"><img src = https://user-images.githubusercontent.com/74092405/134909090-da10c78b-bd03-4073-8e2b-896447f07f1d.png width = 600></p>


- 또한 이를 통해, Dataset을 학습용, 검증용, 테스트용으로 각각 생성하여 디렉토리에 별도로 분리함.
- 학습용, 검증용, 테스트용 image들과 annotation 파일을 지정.
- 이때, annotation파일은 각각 1개씩만 가짐.

