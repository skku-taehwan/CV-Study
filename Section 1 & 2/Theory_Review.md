## Object Localization/Detection/Segmentation

<p align = "center"> <img src=https://user-images.githubusercontent.com/74092405/132098269-769f6012-5001-4260-b2f4-8292a8d0ef50.png> </p>

**1. Object Localization이란?**
- 이미지 속에서 단 하나의 Object(객체)의 위치만을 Bounding box를 통해 찾아내는 것.

**2. Object Detection/Segmentation이란?**
- 이미지 속에서 여러 개의 Object들의 위치를 Bounding box를 통해 찾아내는 것.
- Segmentation이란, Object Detection을 통해 찾은 Object들을 Pixel 단위로 Detection하여 찾는 것을 말함. 

## 일반적인 Object Detection 모델
<p align = "center"> <img src = https://user-images.githubusercontent.com/74092405/132098360-828a48f0-94e8-41de-84d4-2c61be28af9e.png width = 600></p>

**ResNet -> Feature Pyramid Network -> Classification & Regression**

**ㄱ.** 원본 이미지와 Object에 대한 정보를 담은 Annotation 파일 준비 

**ㄴ.** VGG/ResNet 등의 Feature extractor (FE)에 넣어 중요한 Feature들만 추출하여 Feature Map 생성
 - Feature map의 경우, 중요한 feature들만 포함됐긴 때문에 이전 데이터보다 추상화된 형태이며, 채널 수는 증가함

**ㄷ.** Feature Map을 Fully connected (FC) layer에 넣어 학습을 진행하고, Classification과 Regression을 진행 

## Object Detection 기법
### 1. Sliding Window
<p align = "center"> <img src = https://user-images.githubusercontent.com/74092405/132098731-6d96919e-81a8-4abc-b345-f4b2677f95a7.png width = 600></p>

 Sliding Window 방식이란, 주어진 이미지를 왼쪽 상단부터 오른쪽 하단까지 순차적으로 윈도우를 이동하면서 Object Detection을 진행하는 것이다.
 예를 들어, 위 오드리 햅번의 사진처럼 사진의 최상단부터 최하단까지 윈도우를 이동하면서 모든 화면을 탐색하여 이미지 내의 Object를 찾는 방법이다.
 
 **윈도우를 탐색하는 방법**
 > **1.** 다양한 형태의 Window를 사용
 > - 사각형의 크기를 늘리거나 줄여서 사용하기도 하며, 정사각형 직사각형 등 다양한 도형을 사용한다.
 > **2.** Scaling 활용
 > - 윈도우의 크기를 변형하지 않고, 이미지 자체의 Scale을 변형하여 여러번 Sliding하여 Object Detection을 진행한다.
 > 
 **한계점:** Object Detection 초기 기법으로, Object가 없는 영역도 무조건 슬라이딩하여 수행 시간이 길고 성능도 상대적으로 떨어짐.

### 2. Region Proposal 
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132098944-9a6924f8-5fae-41be-bb88-32243664ea23.png></p>
영역 추정 방식으로, 원본 이미지에서 Object가 있을만한 위치(또는 영역)를 Bounding box로 모두 표현하고, 최종 후보를 도출하는 방식이다.
대표적으로, Selective Search 기법이 있다.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132099130-15427246-9f82-4de1-b4f3-eebb087d019a.png></p>

**Selective Search의 알고리즘**
> **1.** 개별 Segment된 모든 부분들을 Bounding box로 만들어 Region Proposal 리스트로 추가
> 
> **2.** 유사도가 비슷한 Segment들끼리 그룹핑
> 
> **3.** 다시 1번 Step Region Proposal 리스트를 추가하고, 2 -> 1 -> 2 프로세스를 반복.

**Selective Search의 장점**
> **1.** 빠른 Detection과 높은 Recall 예측 성능을 동시에 만족
> 
> **2.** 색깔, 무늬, 크기, 형태에 따라 유사한 Region을 **계층적 그룹핑 방법**으로 계산
> - 즉, 비슷한 Object의 영역들끼리 하나로 합쳐서 표현함.
> 
> **3.** 최초에는 **Pixel Intensity**에 기반한 graph-based segment 기법에 따라 Over Segmentation을 수행함.
