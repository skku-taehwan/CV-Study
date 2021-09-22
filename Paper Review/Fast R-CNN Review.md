# Fast R-CNN Paper Review

### R-CNN
- R-CNN 모델은 이미지 분류 분야에서 처음으로 딥러닝을 적용한 모델이며, 그 과정에서 하나의 이미지로부터 2000개의 Region Proposal을 생성함.
- 그리고, 이를 모두 같은 크기로 변환하여(warping) CNN 네트워크 Input으로 하여 학습을 진행하기 때문에 학습 시간이 매우 오래 걸리고, 성능이 떨어지는 단점이 있음.

### SPPnet
- 이런 R-CNN 모델의 단점을 극복하기 위해 제안된 것이 SPPnet으로, Spatial pyramid pooling (SPP) layer를 활용함.
- SPP는 서로 다른 이미지의 크기를 고정된 크기로 변환하는 기법으로, 이미지의 영역을 쪼개서 각각의 Feature에 대한 속성값을 히스토그램으로 저장하는 Spatial pyramid matching (SPM)을 사용함.
- SPM을 통해, 서로 다른 크기의 Feature map을 균일한 크기의 vector로 표현할 수 있음.
- 이를 R-CNN에 적용하여 이미지 하나 당 2000번의 CNN을 통과하던 모델이 단 한 번만 통과하는 모델이 됨.
- 따라서, R-CNN에 비해 획기적으로 학습 시간을 단축했다고 볼 수 있음.


## Fast R-CNN
- 위 논문은 기존의 R-CNN과 SPPnet 알고리즘의 성능을 개선한 모델로, Object detection부터 Classification과 Regression 단계 모두 Deep learning framework로 전환하여 End-to-end learning이 가능토록 하였음.

### Abstract
> Fast R-CNN은 deep convolutional network를 사용하여 object가 있을만한 위치를 효율적으로 분류하는 모델임.
>
> 또한, 학습 및 검증 속도를 획기적으로 향상시켰음.
> 
> - R-CNN으로 VGG16 network 학습의 9배 검증의 213배, PASCAL VOC 2012의 mAP보다 높은 값을 얻음.
> 
> - SPPnet으로 VGG16 network 학습의 3배, 검증에 10배, 그리고 더 정확함.

### Introduction
> **Object Detection 분야의 문제점**
> 
> - Object가 있을만한 후보가 너무 많음.
> 
> - 후보들이 정확한 위치가 아닌 대략적인 localization만 제공함.
>
> -> 즉, 더 빠르면서 정확성이 뛰어나고 간단한 object detection 모델이 필요함.
> 
> **해당 논문에서는 Single-stage training algorithm을 제안하고자 함.** 

**1.1. R-CNN and SPPnet**
#### R-CNN의 문제점
> **1.** Multi-stage pipeline으로 학습
> 
> - R-CNN은 logloss를 활용하여 ConvNet을 finetuning함. 그후에 SVM을 통해 fitting을 진행.
> 
> - 이러한 SVM을 **softmax**로 대체하였음.
> 
> **2.** 학습 시간과 공간이 많이 필요함.
> 
> - 2000번 CNN을 지나고, 그때마다 해당 변수와 값을 저장해야 하기 때문 (R-CNN)
> 
> **3.** Object detection이 느림.

#### SPPnet의 문제점
> R-CNN과 마찬가지로 multi-stage pipeline을 가짐. (Fetaure extraction, fine tuning, SVM, Bbox regression 등)

이러한 문제점을 해결하기 위해 Fast R-CNN을 제안함.

### Fast R-CNN architecture
1. 전체 이미지를 Input으로 사용함. (Object proposals의 집합)
2. Conv layer와 Pooling layer로부터 Feature map (F.M.)을 추출함.
3. F.M.으로부터 각가의 object proposal에 대해 ROI pooling layer로 Fixed-length feature vector를 추출함.
- 이때, feature vector는 2개로 나뉘는데, 첫 번째는 softmax probability 계산을 위한 k개의 obejct class와 background를 의미함. (총 k+1개의 class)
- 두 번째는 Bounding box regression을 위한 k개의 object의 Bbox position에 대한 정보 (좌푯값)

#### ROI pooling layer
- Region of interest (ROI) pooling이란, feature map 상의 임의의 ROI를 지정하여 그에 따라 이미지를 탐색하고, 매핑을 진행함.
- 주로, Max pooling을 적용하여 이미지로부터 의미 있는 정보를 추출함. (보통 7x7 크기의 fixed spatial window를 사용)

#### Initializing from pre-trained networks
학습을 위해 pre-trained ImageNet network를 사용.
1. 마지막 max pooling layer를 ROI pooling layer로 변환하여 고정된 크기의 feature vector를 얻도록 함.
2. Networ의 마지막 FC layer와 softmax layer에 bounding box regression layer를 추가함.
3. CNN의 input으로 이미지의 list와 그 이미지들에 대한 ROI list를 넣어줌.

#### Finetuning
학습 진행 중, Finetuning은 Multi-task loss와 SGD를 통해 진행함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/134299457-fa335a08-dbf4-40c2-804a-f7803aecb3b7.png width = 600></p>

- 해당 식을 통해 Classifiaction과 Regression 부분의 loss를 동시에 업데이트함.
- p의 경우 각각의 ROI별로 softmax를 통해 해당 class에 속할 확률 분포이며, t의 경우 모델을 통해 예측한 bounding box의 좌표.
- u는 ground truth의 class를 말하며, v는 ground truth bounding box regression target을 의미함.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/134299312-15ecded5-1246-4732-af63-49332f069f11.png width = 600></p>

- mini-batch를 128로 설정하였으며, IoU 값이 0.5보다 크면 positive가 되도록 설정하였음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/134300097-e2d7397f-a2e4-4213-a139-6788cee36de9.png width = 600></p>

- 위 식을 통해 ROI pooling layer의 back-propagation을 진행.
- 이때, y_rj는 r번째 ROI의 j번째 output을 의미함.

- 또한, hyper-params 튜닝에 Stochastic gradient descent (SGD)를 사용하며, object detection 시간을 줄이기 위해 Truncated SVD를 사용함.



이를 통해 학습을 진행하였고, 결과는 아래 표와 같음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/134300441-e3b479d4-3346-422c-b449-de73bce4b13a.png width = 600></p>


#### 학습 데이터
VOC 2007, 2010 & 2012

#### 모델
1. CaffeNet - **S**
2. VGG_CNN_M_1024 - **M**
3. VGG16 model - **L**

**Fast R-CNN, R-CNN, SPPnet 학습, 검증 시간 및 성능 비교
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/134301152-7b538a87-7979-4091-9720-66c3443c060d.png width = 600></p>

#### Main result (Conclusion)
1. State-of-the-art (SOTA) on VOC07, 2010, and 2012
2. Fast training and testing compared to R-CNN, SPPnet
3. Fine-tuning conv layers in VGG16 improves mAP




