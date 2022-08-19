# 이미지열 예측을 통한 fps증강

1. **개요** 

   딥러닝을 이용해 영상의 각 프레임 사이에 적합한 이미지를 예측,

   fps증강을 통해 보다 부드러운 영상출력을 끌어내기 위함이다.

   

2. **과정**

   - 데이터 수집

     [**Youtube**](https://www.youtube.com)에서 키워드를 기반으로 URL 추출해서 영상 다운로드

     → 영상을 프레임 단위의 이미지로 추출

     <img src="https://github.com/durudumba/Fps_augmentation/blob/main/img/data.PNG">

     

   - AutoEncoder 네트워크 구현 및 학습

     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/net.PNG">

     이미지 데이터를 학습에 적합하게 변환(크기조정, 수치화)

     Learning rate : 1e-4 / Epoch : 3 / Loss : MSE / Optimizer : Adam

     

   - 테스트

     5분 20초 길이의 영상(256x144, 30fps) → 9,607장의 이미지(256x144)

     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/test.PNG">

     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/result.PNG">


3. **한계**

   프레임 사이 예측이미지에 격자무늬가 생겨 어두운 형태를 띈다. 문제점을 해결하기 위해 정규화 제거, 패딩, 네트워크 변경 등을 시도해봤으나 개선할 수 없었다.

