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
 > 
 > **2.** Scaling 활용
 > - 윈도우의 크기를 변형하지 않고, 이미지 자체의 Scale을 변형하여 여러번 Sliding하여 Object Detection을 진행한다.
 > 
 **한계점:** Object Detection 초기 기법으로, Object가 없는 영역도 무조건 슬라이딩하여 수행 시간이 길고 성능도 상대적으로 떨어짐.

### 2. Region Proposal 
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132098944-9a6924f8-5fae-41be-bb88-32243664ea23.png></p>
영역 추정 방식으로, 원본 이미지에서 Object가 있을만한 위치(또는 영역)를 Bounding box로 모두 표현하고, 최종 후보를 도출하는 방식이다.
대표적으로, Selective Search 기법이 있다.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132099130-15427246-9f82-4de1-b4f3-eebb087d019a.png width = 600></p>

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


### 3. NMS (Non max supression)

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112019-a9730c69-1edb-4d45-bc25-5e3ac417c7e8.png width = 600></p>

- Object Detection 알고리즘은 Object를 놓치면 안 되기 때문에 Object가 있을만한 모든 위치에 Detection을 수행하는 경향이 강함.
- NMS는 감지된 모든 Object의 bounding box 중에서 비슷한 위치에 있는 box를 제거하고, **가장 적합한 box를 선택하는 기법**.
- 즉, 가장 확실한 bounding box 외의 나머지 박스들은 모두 제거하는 기법임.
- 위 사진처럼, 자동차 주변에 많은 bounding box가 생성되어 있는데, 하나의 차를 인식할 수 있는 가장 확실한 bounding box만 남겨두고, 나머지는 삭제.


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112127-d886871a-27b1-4da0-9372-a3afc72d3787.png width = 600></p>

**NMS 수행 알고리즘**

> **1.** Detected된 bounding box별로 특정 Confidence threshold값을 계산하여 특정값 이하의 bounding box는 제거
> 
> **2.** 가장 높은 confidence score 값을 갖는 box 순으로 내림차순 정렬 후, 그 box와 겹치는 다른 bounding box들과의 IoU값을 계산하여 특정 threshold 이상인 box는 모두 제거함.
> 
> **3.** 남아있는 box만 선택
> 
> - 즉, Confidence score가 높을수록, IoU threshold가 낮을수록 많은 bounding box들이 제거됨.


## Object Detection 성능 평가 Metric

### 1. IoU (Intersection over union)

- Object Detection 모델이 예측한 결과(Bounding box)와 실측(Ground truth box)가 얼마나 일치하는지를 나타내는 지표

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132111921-09fb6917-4ad6-45bf-972b-47e03f2e1746.png width = 600></p>

- 위 그림처럼 실제 object가 있는 위치와 예측한 bounding box를 포함한 영역과 두 박스가 일치하는 영역의 비율로 IoU를 계산함.
- IoU 값에 따라 Detection 성능이 좋다/나쁘다를 판단할 수 있음
- 하지만, 그 값에 대한 기준이 대회/데이터에 따라 다름

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112014-885f8610-0ca4-4aa4-85eb-e921d76db46a.png width = 600></p>

> **1.** Pascal VOC: IoU Metric이 0.5이하이면 False라 판단하고, 그 이상이면 True라 판단.
> 
> **2.** MS CoCo: 다양한 IoU threshold 기준을 적용하여 평가하는데 있어 까다로움.

### 2. mAP (mean average precision)
- 실제 Object가 Detected된 재현율(Recall)의 변화에 따른 정밀도(Precision)의 값을 평균한 성능 수치.

#### 정밀도(Precision)와 재현율(Recall)


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112232-8e1557b2-de5d-4dda-bf6a-ba1c3a54f958.png width = 600></p>


> **1. 정밀도(Precision):** 예측을 Positive로 한 대상들 중에 예측과 실제값이 Positive로 일치하는 데이터의 비율
> 
> - Object Detection에서는 검출 알고리즘이 검출 예측한 결과가 **실제 Object들과 얼마나 일치**하는지를 나타내는 지표
>
>  ex) 이미지에 있는 동물들을 새(bird)라고 예측했을 때, 실제로 새(bird)일 확률
> 
> **2. 재현율(Recall):** 실제값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치하는 데이터의 비율
> 
> - Object Detection에서는 검출 알고리즘이 **실제 Object들을 빠뜨리지 않고, 얼마나 정확히 검출 예측**하는지를 나타내는 지표
>
>  ex) 이미지에 있는 새(bird)들을 몇 마리나 정확히 검출했는지의 비율
>  
>  **3. 오차 행렬(Confusion Matrix)**
>  
>  <p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112285-39a2dcab-0d33-49ca-9fb2-6019c4be8af3.png width = 400></p>

#### Confidence 임곗값에 따른 정밀도-재현율 변화


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112319-4355094a-8883-41b8-b505-6752e34d8bde.png width = 600></p>

Confidence 임곗값이 낮을수록 더 많은 bounding box를 생성하여 정밀도는 낮아지고, 재현율은 높아짐.
- 즉, 이미지의 아무 곳에나 bounding box를 난사하여 실제 object를 찾기가 어려워짐

반대로 Confidence 임곗값이 높으면, bounding box를 잘 만들기 않기 때문에 정밀도는 높아지고, 재현율은 낮아짐.

### 정밀도 재현율 곡선 (Precision-recall curve)
- Recall 값의 변화에 따른 Precision값을 나타낸 곡선.
- 이때, 해당 그래프 아래의 면적을 AP (Average precision)이라고 부름.

![image](https://user-images.githubusercontent.com/74092405/132112528-0c9c5e22-6d97-4964-a2dc-b71576a619a4.png)


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/132112319-4355094a-8883-41b8-b505-6752e34d8bde.png width = 600></p>
