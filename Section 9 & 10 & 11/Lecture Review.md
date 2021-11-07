## RetinaNet
### RetinaNet의 이전의 Object Detection
- RetinaNet이 출현(2017년)하기 이전에는 One-stage detection model의 Inference 속도는 빠르지만, 그 성능이 Faster RCNN보다 뛰어나지 않았음.
- RetinaNet은 One-stage detector의 빠른 Detection 속도와 동시의 Detection 성능이 저하되던 문제를 개선함.
- 수행 시간이 YOLO나 SSD (Single shot detector)보다는 느리지만 Faster RCNN보다는 빠름.
- 특히, 다른 One-stage detector보다 **small object**에 대한 detection 능력이 뛰어남.
- 주요한 특징으로는, Focal Loss를 적용한 FPN (Feature pyramid network)를 사용.


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140632933-7ddc2486-fdda-4e18-a6bc-5ff61d155a2e.png width = 800></p>

### Focal Loss
**1. Cross Entropy**
<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140632998-d7560d91-457c-4eaa-8efb-91a782daaa01.png width = 800></p>

- 기존의 FPN은 Cross entropy를 통해 Loss를 계산했기 때문에 학습 중 사용되는 anchor box들이 상대적으로 큰 Object나 Background(Negative sample)에 치중하여 학습하는 경향이 있음.
- 즉, 이미 잘 detection되는 object들을 더 잘 찾는 방향으로 학습되는 Class imbalance 문제가 발생.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633050-05072986-7ee0-4e72-8453-8bf7ee576d1d.png width = 800></p>

- **Easy example**: 찾기 쉬운 대상들, Background나 크고 선명한 object. 이미 높은 예측 확률을 가짐.
- **Hard example**: 찾기 어려운 대상들, 작고 형태가 불분명하여 낮은 예측 확률을 가짐.

- Easy example이 많고, hard example이 적은 class imabalnce 이슈는 Object detection 분야에서 지속되어 왔음.
- 특히, Bbox regression과 classification을 각각 진행하는 two-stage etector의 경우, RPN (Region proposal netwrok)을 통해 object가 있을만한 곳을 높을 확률 순으로 먼저 필터링함.
- 하지만, One-stage detector는 Region proposal과 많은 object 후보들 중에서 detection을 같이 수행하므로 Class imbalance로 인한 성능 저하 영향이 큼.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633175-a5e94692-51c5-4138-b0e2-7deef3507681.png width = 800></p>

- 또한, Easy example에 대해 더 잘 detection하기 위한 방향으로 학습이 진행되다 보니, hard example의 loss보다 easy example의 loss가 커지는 현상이 발생함.
- 이로 인해, Background나 확실한 object에 대해서는 더 정확한 예측이 가능하지만, 작은 object들에 대한 예측은 어려워짐.


<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633221-dfa41a43-c13b-45a8-a00e-8fdad7d8d1a6.png width = 800></p>

- 따라서, 이러한 Class imabalance를 해결하기 위해 기존의 One-stage detector 모델에서는 foreground & background를 적절히 섞거나, 작은 object들을 crop하거나 확대하는 등의 샘플링을 진행.
- 하지만, RetinaNet에서는 샘플링 기법이 아닌 Cross entropy 자체를 개선할 방법을 찾았고, Focal loss가 그 방법임.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633289-e8c915f5-0f2c-454c-b72a-f55dff33d3c7.png width = 800></p>

- Focal loss란 cross entropy loss에 가중치를 부여하여 Detection하기 쉬운 object일수록 loss 값이 작게 하고, 어려운 object일수록 loss 값을 상대적으로 크게 하여 골고루 학습을 진행하도록 함.


### FPN (Feature pyramid network)
- 또한, FPN을 함께 적용하여 추상화 정도가 다른 여러 level의 Feature map들을 연결하여 object detection 성능을 높임.
<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633330-461eca05-753e-4fc7-8cb4-06c2995ee46e.png width = 800></p>

- 기존의 Sliding window 기법은 큰 object detection이 가능하지만, 컴퓨팅 시간과 detecting 시간이 오래 걸림.
- 이를 해결하기 위한 것이 FPN으로, F.M의 특성들이 상위(Bottom up)로 갈수록 소실되는 현상을 방지함.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633745-7b5df9c2-d5b5-406d-99d7-be20d6ee059b.png width = 800></p>

- FPN은 Top-down 방식으로 작동하며, ResNet에서는 상위 feature map을 2x upsampling을 통해 크기를 확대시키고, 1x1 conv를 지난 하위 feature map과 merge시킨 후, class subnet을 통해 예측을 진행.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633806-053597bc-2e22-457f-af1b-cd4f84966903.png width = 800></p>

- ResNet FPN에서는 상기에 설명한 Lateral connection을 통해 Top down merge를 진행.
- Merge 이후의 3x3 Conv block은 2개의 서로 다른 signal이 섞이면 본래 가지고 있던 자신만의 특성을 잊어버리는 현상을 방지하기 위함.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633902-37fef384-9f37-4748-a624-b53991557c0d.png width = 800></p>

## EfficientDet
<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140633979-d3296311-3d86-478f-b818-da961154fc4b.png width = 800></p>

- EfficientDet은 EfficientNet을 Backbone으로 사용하고, BiFPN을 Neck, Classification과 Bbox regression을 Head로 하는 모델임.
- 또한, Backbone, neck, head에 Compound scaling을 통해 최적의 모델 구조를 찾는 특징을 가짐.
- 적은 연산 수와 파라미터 수에 비해 상대적으로 높은 모델 예측 성능을 보여줌.

### EfficientNet
<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140634024-b0c72f51-830b-4459-bd9a-4082ce77c033.png width = 800></p>

- EfficientNet은 네트워크의 깊이(Depth), 필터 수(Width), 이미지 Resolution 크기를 최적으로 조합하여 모델의 성능을 극대화 시키는 모델임.
- 각각의 요소들을 따로 최적화해서는 최적의 모델을 구축할 수 없기 때문에 최적의 조합을 찾기 위한 연구를 통해 개발됨.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140634069-17305bb0-9ad8-4aaf-b106-10107f00ff85.png width = 800></p>

- 자동차를 예를 들면, 마력이 클수록, 타이어나 휠의 성능이 좋을수록, suspension이 뛰어날 수록 차의 성능이 좋은 것처럼 모델의 필터 수/네트워크 깊이/resolution도 그러한 조합이 있을 것임.
- 각각의 요소들을 고정시키고, 한 가지 요소를 변화시키면서 모델의 성능을 살펴보아도 특정값 이상의 성능 향상 효과를 얻기는 어려움.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140634089-2782062f-b2b2-4ea5-981e-4a13d40dfa89.png width = 800></p>

- 따라서, 이러한 요소들에 관련된 최적화 함수를 설정하여 최적화를 진행.

<p align = "center"><img src = https://user-images.githubusercontent.com/74092405/140634129-6a0425e2-dc01-49ac-b66d-eaeb809c28e9.png width = 800></p>


- 이를 통해, B0~B7까지의 모델을 구축.
- Inference 속도가 빠르면서 동시에 예측 성능도 향상됨.










