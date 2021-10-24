# SSD (Single Shot Detector)
1. SSD가 개발된 당시까지, inference 속도와 성능 모두가 좋은 모델이 없었음.
2. 성능이 좋으면서 Inference 속도가 빠른 Faster-RCNN의 경우에도 최대 7 FPS로 그리 빠른 편은 아니었음.
3. 또한, YOLO v1의 경우에도 inference 속도는 빠르지만, 그 성능이 좋지는 못 했음. (최근 버전은 속도와 성능 모두 뛰어난 편)


<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951248-1fa2b274-0185-48a0-bbb8-01ffba8b8f7f.png width=600></p>

- SSD는 Faster-RCNN 계열의 Two-stage detector가 아닌 One-stage detector의 본격적인 연구를 이끌어온 모델임.
- 2016년 기준으로 SOTA를 기록하였음.

## SSD Network Architecture

<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951348-4f254644-cded-4f6f-8ea1-fb4feaff18a8.png width=600></p>

- 입력되는 원본 이미지의 크기는 300x300, 512x512 중 하나를 선택할 수 있음.
- Convolution layer가 반복되는 VGG-16 Network를 통해 Feature map을 생성함.
- Multi-scale feature layer를 사용하는데, 이때 faster-rcnn에서 사용했던 anchor box와 비슷한 default box를 활용함.

### Multi-scale feature layer
<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951506-2a912ce9-2c39-487e-9d0a-6232a329a9e1.png width=600></p>

- 기존의 object detection을 위해 사용되던 sliding window 기법은 이미지의 크기를 조정하면서 같은 크기의 window를 사용하여 object detection을 진행하는 Image pyramid를 사용하였음.

<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951618-248a0d59-4908-43d7-abc2-d864b79023f4.png width=600></p>

- 이와 달리, SSD에서는 Feature map의 크기를 조정하면서 object detection을 진행함.
- 원본 이미지에서 추출한 Feature map의 크기를 줄여나감으로써, 작은 object부터 큰 object까지 detection을 진행하고 여기에서 도출된 feature들을 합쳐서 학습을 진행함.

<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951806-ce07f8af-e0c9-4837-b5d4-6fb1596e2fba.png width=600></p>

- 이는, Anchor box를 사용하는 Faster-RCNN의 RPN과 유사함.

<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135951915-55346a06-c378-43f6-befc-17ad2eefe235.png width=600></p>

- 위와 같이 원본 이미지로부터 feature map을 뽑아낸 후에 anchor box를 이용하여 각각의 object별로 softmax 값을 계산하여 학습을 진행함.
- 이러한 방법과 유사하게, SSD는 default box를 활용하여 학습을 진행.

### SSD Network
<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135952030-c0f2ff8f-e378-4728-a1d8-938753cc248a.png width=600></p>

- 모든 convolution layer에서 3x3의 filter를 사용하며, default box의 aspect ratio가 각각 4개, 6개, 6개, 6개, 4개, 4개를 사용하여 총 8732개의 object detection box를 생성함.
- 그리고 이들에 대한 softmax값을 구하고, NMS를 활용하여 confidence score가 높은 값들을 추출함.

<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135952273-35504a51-fb1d-44dd-8160-600f5452ce42.png width=600></p>

- SSD Network에서 multi-scale feature map과 default box를 적용 결과로, 큰 feature map에서는 작은 object를 잘 detect하고, 작은 feature map에서는 큰 object를 잘 detect함.

### Loss function
<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135952367-536505bc-db1b-44a4-9fca-15e1dd40d90e.png width=600></p>

- Classification과 Bbox regression에 대한 loss를 함께 학습함으로써, ground truth와 default box 간의 offset을 최소화하는 방향으로 학습을 진행함.
- 또한, data augmentation 기법을 사용하면 그 성능을 더욱 높일 수 있어 해당 논문에서는 random crop, 이미지 확대 및 축소 등의 방법을 사용하였음.
<p align = "center"><img src= https://user-images.githubusercontent.com/74092405/135952491-52fe5f73-18dc-43de-98f8-b298ec5e0844.png width=600></p>



