# Lecture Week 2

## R-CNN (Regions with convolutional networks features)
1. Region Proposal (Selective search 기법 등) 방식을 활용하여 Object Detection을 진행하는 RCNN.
- Object가 있을 만한 영역을 예측함.
2. 예측된 Object들을 CNN 모델에 넣어 학습을 진행.
- 설정한 Region들을 CNN의 feature로 활용하여 Object detection을 수행하는 신경망.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132991808-d5dac56d-45d3-48a1-a571-1d5fc15be7bc.png width = 600></p>

- 예측된 Region bbox들(혹은 이미지)을 같은 크기로 변환한 후, CNN에 넣어주기 때문에 원본 Object가 찌그러지는 현상이 발생함.
- 이는 RCNN 모델의 Classification dense layer에 입력될 때, 이미지 크기가 동일해야 하기 때문임.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992035-c2eaa10b-dd60-4bc3-b5fa-db3de547b435.png width = 600></p>

1. 이미지를 삽임 -> Region proposal 진행.
2. 추출된 object들을 동일한 크기로 변환. (warp or crop)
3. CNN에 입력하여 학습을 진행. (Feature extractor -> Feature map -> Flatten fully connected layer)
4. SVM Classifier를 이용한 분류 진행 + Bounding box regression 진행
- 단점: 이미지마다 Region proposal로 인한 2000개의 예상 Object 위치를 가진 이미지가 나오기 때문에 학습 시간이 오래걸리며, Inference 시간도 오래 걸림. 대략 한 이미지당 50초의 Object detection 시간이 소요됨.
- 장점: 성능이 1 Stage detector에 비해 뛰어남. Object Detection 분야에서 처음으로 딥러닝을 적용시킨 기법임.

### R-CNN의 학습 - Classification
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992371-4d735f56-0c77-421d-b975-2890ebcb85e1.png width = 600></p>

- SVM Classifier 직전까지가 CNN Network임.
- GT (Ground truth)로 학습을 진행시키며, IoU값이 0.3 이하이면 Bg (Background)로 설정.
- 0.3 이상이지만 GT가 아닌 경우에는 학습에서 제외시킴.

### Bounding box regression

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992476-9889c5ba-8f7c-4ab8-8f83-99b239a6e080.png width = 600></p>


- 초기에 y로 입력했던 레이블의 값과 CNN을 통해 예측된 Bounding box 간의 차이를 줄이도록 학습을 진행.
- 실제로 우리가 CNN을 통해 구하는 값을 dx(p)와 dy(p)이며, 각각의 예측값이 실젯값과 같아지는 것이 이상적임.
- 그림의 손실 함수를 최소화하는 것이 RCNN 알고리즘의 목적.


## SPP (Spatial pyramid pooling) Net
### RCNN의 문제점

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992693-f77aed6e-c54a-4813-b000-caa45a3946e2.png width = 600></p>


**1.** 2000개의 Region 이미지가 CNN으로 입력되면서 학습(object detection) 시간이 상당히 많이 소요됨.
> CNN이 다른 크기의 이미지를 수용하지 않음.
> 
> Flatten fully connected layer의 input 크기가 고정되어야 하기 때문.
> 
>  -> Region proposal 이미지를 SPP Net의 고정된 크기 Vector로 변환하여 FC에 1D Flattened된 input 제공.

**2.** Region 영역 이미지가 Crop/warp되면서 원본과 달라지는 현상 발생.
> 2000개의 Region proposal 이미지를 feature extraction하지 않고, 원본 이미지만 CNN으로 F.M.을 생성한 후에 원본 이미지의 selective search로 추천된 영역의 이미지만 F.M.으로 매핑하여 별도 추출.
> 
> 즉, 2000개의 예측 이미지 대신 원본 이미지 1개만 사용하여 F.M.을 만들자는 의견.

### SPP (Spatial pyramid pooling) 이란?
- CNN Image classification에서 서로 다른 이미지의 크기를 고정된 크기로 변환하는 기법.


<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992728-45bf2018-cb68-4e3c-9676-a1863bd253c0.png width = 600></p>

- 즉, Feature map과 Dense layer 사이에 SPP layer를 삽입하여 둘 사이를 유연하게 연결해줌.

#### Bag of visual words
- NLP 영역의 Bag of words와 비슷하게, 하나의 object를 이루고 있는 다양한 요소들을 data화 시킨 것.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992778-a18ad607-461b-488e-8254-e5ac906f4151.png width = 600></p>

- 원본 이미지를 잘게 쪼개서 새로운 Mapping 정보를 생성함.
- 이때, Histogram을 통해, mapping된 object들의 값을 얻음.
- 즉, 비정형 data를 정형 data로 변환함. 

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992919-ad7c8c70-9406-4895-a2f7-b66f2e633215.png width = 600></p>

> But, 이미지의 일부만으로 전체 이미지를 판단할 수 없음.
> 
> 예를 들면, 위의 그림처럼 모래나 하늘만 보고, 이미지 속 장소가 바다라고 판단하기는 어려움.
> 
> 따라서, 각 위치마다 어떤 특성을 가지고 있는지도 포함되어야 함.

#### SPM (Spatial pyramid matching) 개요
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992941-ab15049f-c145-49c7-9259-9fecc678707b.png width = 600></p>

- Bag of visual words의 문제점을 해결하기 위해, 영역에 따른 object의 위치별 히스토그램을 획득.
- 새로운 Feature에 대한 속성값을 얻어 분류에 적용함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132992974-e00a45dd-09d2-43c7-a2f7-75afa90a7fe7.png width = 600></p>

- 이를 통해, 서로 다른 크기의 Feature map을 균일한 크기를 갖는 Vector로 표현할 수 있음.
- Ex) feature가 3개인 경우, 3x1 + 3x4 + 3x12 = 63개의 원소를 갖는 Vector 값으로 feature들을 표현할 수 있음.

#### SPP (Spatial pyramid pooling)
**Pooling이란:** 원본 feature에서 특정 크기만큼을 추출하는 기법을 말함.
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132993084-ef6a7d33-d410-43a5-8114-d8dd9f7f3281.png width = 600></p>

- SPP는 SPM을 이용하여 얻은 Histogram들 중, Max pooling을 통해 영역별로 가장 큰 값만을 추출함.
- 즉, 위의 예시에서 1x1 + 4x1 + 16x1으로 모든 feature를 표현.

### SPP Net
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132993193-64ae251c-d85b-405e-8453-a68a96c12b90.png width = 600></p>


따라서, 위의 SPP 알고리즘을 활용해 동일한 크기의 Vector로 만드는 기법을 SPP Net이라고 함.
- 서로 다른 feature map의 사이즈를 동일한 크기로 변환하여 flatten하게 만들어줌.
- 이를 통해, Pooling 후 발생하는 손실을 줄여 이미지에 대한 분류 성능을 향상시켰음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132993242-740a0839-1ad0-459d-a7a4-8f8e4d46d874.png width = 600></p>

- 이를 RCNN에 적용하여, 기존 RCNN의 2000개의 Region proposal을 CNN에 넣지 않고, 원본 이미지만을 CNN을 통과시킴으로써 학습 시간을 단축시킴.

## Fast RCNN

### R-CNN의 발전

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133008861-255db58d-ccbd-4550-b0c2-45a850c9ba89.png width = 600></p>

**1. R-CNN:** 처음으로 Object detection 분야에 DL을 적용하여, Classification 전 단계에 CNN을 도입.
**2. Fast R-CNN:** 이미지 1개로만 학습을 진행하고, ROI Pooling을 사용하여, 2000개의 Region proposal을 사용하던 R-CNN의 학습 속도를 개선함.
**3. Faster R-CNN:** Selective search 기법을 DL network 안에 포함시켜 RPN구조로 만들어 사용.

### Fast RCNN

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133008901-0ac2c2c8-b79c-4163-9d0d-ebb0a638b065.png width = 600></p>

- SPP Net의 SPP layer를 ROI pooling layer로 사용함.
- **SPP Net**은 입력된 이미지 자체를 Convolution network에 넣어 feature map을 만들고, 그 F.M.으로부터 Region proposal을 생성함.
- ROI pooling을 통해 ene-to-end 학습을 가능하게 함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009227-05a603c6-ef3f-4994-981e-dc2071ac40c7.png width = 600></p>

- ROI pooling이란 ROI (Region of interest) 영역에 해당하는 부분을 max pooling하여 feature map을 고정된 길이의 저차원 벡터로 축소함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009294-080f8202-5314-4e29-a1ee-57e2013aab30.png width = 600></p>


- Fast RCNN은 CNN feature extraction, classification, bounding box regression을 모두 하나의 모델에서 학습시키는 모델임.
- 이때, Multi-task loss 함수를 적용함. (Classification loss + Regression loss)
- 또한, Classification 단계에서 SVM classifier 대신 softmax 함수를 사용하여 분류를 진행함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009351-f541a39c-7f66-48ea-be55-7bcebc0b70da.png width = 600></p>

## Faster RCNN

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009401-3c5b2166-192b-4652-bd60-fc5192e50476.png width = 600></p>

- Faster RCNN은 위의 그림과 같이 Fast RCNN에 RPN을 추가한 모델임.
- RPN (Region proposal network)은 Region proposal 시에 사용되는 selective search 기법을 neural netwrok 구조(RPN)로 변경함으로써, 학습과 Inference 속도를 향상시켰음.
- Fast RCNN과 마찬가지로, End to end 학습 네트워크를 가짐.
- Object detection을 구성하는 모든 요소가 DL으로 구현된 첫 번째 모델.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009568-f33cc186-c6b7-4019-9130-90f51f3ef225.png width = 600></p>

- Anchor box를 reference로 활용하여 object가 있을만한 영역의 후보 bounding box를 얻음.
- 즉, Ground truch를 기준으로 데이터를 넣어 anchor box를 학습하는 network라 할 수 있음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009618-22274732-b3fc-4798-90b0-6958e5f03777.png width = 600></p>

- Anchor box는 총 9개의 서로 다른 크기와 비율을 같는 사각형 프레임으로, Box의 중심점이 되는 grid point를 기준으로 anchor box를 생성함.
- 서로 다른 크기와 비율의 anchor box로 인해, 하나의 anchor box만으로는 다른 크기의 같은 object를 detection하기 어려움.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009807-cef18a12-8952-47da-9c45-edc1d5157577.png width = 600></p>

- 각각의 grid point로 부터 anchor box를 생성하기 때문에 총 50x38x9개의 anchor box가 생성됨.
- 이렇게 생성된 anchor box를 RPN을 통해 학습을 진행.


### RPN Network 구조
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009904-b8d520d4-c105-4065-ac75-a45f7085fe7c.png width = 600></p>

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133009948-7621c685-d2db-4593-a392-dc82116dd105.png width = 600></p>

- RPN Network는 원본 이미지를 통해 생성된 feature map을 Convolution layer의 입력으로 받아 Class를 분류하는 layer와 bounding box의 위치를 regression하는 layer에서 학습이 진행됨.

**RPN bounding box regression**

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133010094-0cdad578-9faf-4c0c-aa4f-17d19325d797.png width = 600></p>

- RCNN box regression의 구조와 유사하지만, 원본 G.T. 대신에 RPN으로 예측된 G.T.와 Anchor box를 사용한다는 점에서 다름.
- RCNN과 표기 방법이 유사하기 때문에 그 점에 유의해야 함.
- 또한, IoU 값이 0.7 이상이면 Positive anchor box, 0.3보다 작으면 Negative anchor box라 하며, 0.3~0.7 사이의 IoU값을 갖는 anchor box는 학습에서 제외시킴. (filtering out)

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133010217-77cefc9c-fce5-458f-b816-ec7740a901f0.png width = 600></p>

- 따라서, 예측된 bounding box와 positive anchor box 간의 거리(좌표) 차이가 ground truch와 positive anchor box 간 거리 차이와 최대한 같아지도록 regression을 하는 것이 RPN의 목적임.
- 즉, predicted boudning box와 ground truth가 거의 동일해지도록 학습을 진행.

### RPN Loss function
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133010367-d3c18ca9-bd03-4848-8046-d45dc6aa24ea.png width = 600></p>

- Positive anchor box만을 Regression 시키는 것이 식으로 나타나 있음.
- Classification 함수로는 Sigmoid와 Softmax 둘 중에 하나를 사용함.

### Faster RCNN Training
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/133010432-5d0bbb90-7495-4020-b9a5-35c910df60ae.png width = 600></p>

1. 원본 이미지로 얻은 F.M.을 입력으로 사용하여 RPN을 학습
2. Fast RCNN Classification/Regression 학습을 진행.
3. 2번에서 나온 결괏값을 활용하여 RPN을 Fine Tuning 진행.
4. 다시 3번의 결과를 활용해 Fast RCNN을 Fine Tuning.


## 참고자료
1. 딥러닝 컴퓨터비전 가이드, https://www.inflearn.com/course/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%BB%B4%ED%93%A8%ED%84%B0%EB%B9%84%EC%A0%84-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C/dashboard
2. Faster R-CNN, https://welcome-to-dewy-world.tistory.com/110

