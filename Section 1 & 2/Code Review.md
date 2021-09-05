# Code Review

## 1. selective_search_n_iou.ipynb
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132131773-fbf70411-9f9e-40f2-8f46-a1338745e3b8.png width = 800></p>

OpenCV 모듈을 ```import cv2```로 가져와 내장된 함수 ```cv2.imread(파일 경로 + 이름)```으로 불러옴
- 이때, imread() 코드는 RGB 이미지를 BGR 이미지로 자동으로 변환함.
- 따라서, cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 함수를 통해 RGB 형태로 바꿔줘야 함.
- 하지만, 우리가 불러온 사진 파일은 흑백 사진이므로 차이는 없음.


<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132131850-3192ae5c-62db-4af9-b0de-f6d11aad395b.png width = 800></p>

마찬가지로, Selective Search 모듈을 ```import selectivesearch```로 불러와 Region Proposal 정보를 반환시킴.
- scale은 이미지의 크기를, min_size는 객체의 최소 크기를 의미함.
- 변수 regions에는 bounding box들의 좌표와 사이즈에 대한 정보가 담김. 

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132131927-b513e29e-9a35-4687-9b07-12ab525bcc54.png width = 800></p>

- ```cand_rects = [cand['rect'] for cand in regions]```코드를 통해 선택된 bounding box의 좌표값들만 불러올 수 있음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132131963-747e5547-2b60-42f6-b8d3-b2ee6adc88c6.png width = 800></p>

- ```cv2.rectangle``` 코드를 통해 bounding box를 시각화할 수 있음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132131981-e8d49e2b-32a2-4329-ab23-f2df28f9ac84.png width = 400></p>

### IoU 값 구하기

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132022-9f4ad4f3-c4cb-4fab-b45e-750f173463b5.png width = 800></p>

- x1, y1, x2, y2의 좌표값을 구해 Ground truth와 예측한 bounding box 간의 공통된 영역을 구할 수 있음.
- 그리고, ground truth와 bounding box 각각의 영역을 구해서 더하고, union영역을 빼면 전체 면적이 나오므로, IoU 값을 구할 수 있음.


<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132136-465c6fda-8a93-423c-9e23-75861a9604f8.png width = 800></p>

- 각각의 bounding box에 대하여 IoU 값을 구한 후, 그 값이 0.5보다 큰 박스들만 남기고, 그 값들을 표시하는 코드.

### 최종 결과
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132176-2ff0a856-0cf7-490d-b645-1cb4b1275843.png width = 800></p>


## 2. pascal_voc_dataset.ipynb
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132195-6509f70b-23d2-449a-9b80-ce53d85f1159.png width = 800></p>

- Colab은 Linux 커널 기반이기 때문에 윈도우 명령어 앞에 ```!```을 꼭 붙여줘야함.
- ```!mkdir``` 코드로 폴더 생성 가능.
- ```!wget 압축파일 웹 주소```로 압축파일을 다운로드 받을 수 있음.
- ```!tar -svf 압축파일 -C 경로```로 다운로드 받은 압축파일의 해제가 가능.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132253-a9a1747a-90fe-474d-a529-121b975a78e4.png width = 800></p>

- ```!ls 경로 | head -n 5```` 코드로 해당 경로 안에 들어있는 파일 5개의 이름을 볼 수 있음.
 
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132288-8e4cccad-bfaf-494d-b6b0-7d5fedb4f647.png width = 800></p>

- ```cv2.imread(os.path.join(경로, 경로 + 파일 이름)```: os.path.join으로 파일이 있는 경로를 받아올 수 있음.
- ```plt.imshow(이미지가 할당된 변수)```로 불러온 이미지를 시각화 할 수 있음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132367-55609407-5ad8-4272-a1a0-eaa68f39cbf3.png width = 800></p>

- 이미지 파일과 Annotation 파일들의 경로를 가져옴.
- 그 후, ElementTree를 통해서 Annotation 파일들을 파싱해줌.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132398-8b3934b8-5e1f-4687-8f38-1d9b61e6d70e.png width = 800></p>

- ```tree```변수에 annotation.xml 파일을 parsing한 후, ```root```변수에 ```getroot()``` 코드를 통해 node 탐색을 진행.
- filename, size, width, height, bounding box 좌표값에 대한 정보를 각각의 노드를 조회하여 찾음.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132461-abdba387-3c81-4520-bf4f-bd22eb55784b.png width = 800></p>

- 이를 통해, 각각의 클래스에 대한 이름과 바운딩 박스 좌표값 리스트를 만들어줌.

<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132492-f5377cda-49b0-49ce-b222-47bbfcb2631c.png width = 800></p>

- 마지막으로, ```cv2.rectangle```과 ```cv2.putText```를 통해, annotation에서 불러온 bounding box와 object name을 이미지에 표시해줌.

### 최종 결과
<p align = "center"><img src=https://user-images.githubusercontent.com/74092405/132132540-909b7da3-36be-43e4-8ef2-68e89d5f91e9.png width = 800></p>



