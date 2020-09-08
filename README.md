# Object Detection 학습 과정

## 데이터 셋 구성 전에 각 이미지 파일에 대하여 바운딩 박스를 그린 후 xml 형식으로 저장.
``` 
|---- train
|     |---- image
|           |---- 1.jpg
|           |---- 2.jpg
|           |---- ...
|     |---- annotation
|           |---- 1.xml
|           |---- 2.xml
|           |---- ...
``` 

## 데이터 셋 구성(VOC Type) 기준
1. move_image_xml.py : image, annotation 파일을 각각 해당 위치로 옴김
   - ./image ./annotation 폴더로
2. make_names.py : annotation 파일을 불러와 class를 뽑아와 data_names.txt 파일을 만듬
   - 클래스 명만 기록
3. make_image_txt.py : image 파일을 불러와 image.txt 파일을 만듬
   - 이미지 파일 이름만 기록
4. voc_convert.py : image, annotation 파일을 각각 불러와, train.txt 파일로 만듬
   - 형식 : 1.jpg xmin, ymin, xmax, ymax, class number
   -     : 2.jpg xmin, ymin, xmax, ymax, class number
   
``` 
# Dataset 위의 과정 시 다음과 같은 파일이 만들어짐
|---- train
|     |---- data.name.txt
|     |---- image.txt
|     |---- train.txt
|     |---- image
|           |---- ...jpg
|     |---- annotation
|           |---- ...xml
``` 
