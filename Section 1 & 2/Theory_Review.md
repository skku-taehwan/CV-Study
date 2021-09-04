### Object Localization/Detection/Segmentation
![image](https://user-images.githubusercontent.com/74092405/132098269-769f6012-5001-4260-b2f4-8292a8d0ef50.png)


**1. Object Localization이란?**
- 이미지 속에서 단 하나의 Object(객체)의 위치만을 Bounding box를 통해 찾아내는 것.

**2. Object Detection/Segmentation이란?**
- 이미지 속에서 여러 개의 Object들의 위치를 Bounding box를 통해 찾아내는 것.
- Segmentation이란, Object Detection을 찾은 Object들을 Pixel 단위로 Detection하여 찾는 것을 말함. 

**일반적인 Object Detection 모델**
![image](https://user-images.githubusercontent.com/74092405/132098360-828a48f0-94e8-41de-84d4-2c61be28af9e.png)

ResNet -> Feature Pyramid Network -> Classification & Regression
> ㄱ. 원본 이미지와 Object에 대한 정보를 담은 Annotation 파일 준비 
>
> ㄴ. VGG/ResNet 등의 Feature extractor (FE)에 넣어 중요한 Feature들만 추출하여 Feature Map 생성
> - Feature map의 경우, 중요한 feature들만 포함됐긴 때문에 이전 데이터보다 추상화된 형태이며, 채널 수는 증가함
> 
> ㄷ. Feature Map을 Fully connected (FC) layer에 넣어 학습을 진행하고, Classification과 Regression을 진행 
