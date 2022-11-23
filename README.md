# 이미지열 예측을 통한 FPS증강

1. **개요** 

   AutoEncoder를 이용해 영상의 각 프레임 사이에 적합한 이미지를 예측,

   FPS증강을 통해 보다 부드러운 영상출력을 끌어내기 위함이다.

   

2. **과정**

   - 데이터 수집

<<<<<<< HEAD
     **Youtube**에서 키워드를 기반으로 URL을 추출해서 영상 다운로드
=======
     [**Youtube**](https://www.youtube.com)에서 키워드를 기반으로 URL 추출해서 영상 다운로드

     → 영상을 프레임 단위의 이미지로 추출

     <img src="https://github.com/durudumba/Fps_augmentation/blob/main/img/data.PNG">
>>>>>>> ce76a6a5a713f73f59eaaee71180be1e9b806dc0

     

   - 데이터 전처리

<<<<<<< HEAD
     다운받은 영상을 프레임 단위의 이미지로 분할
=======
     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/net.PNG">
>>>>>>> ce76a6a5a713f73f59eaaee71180be1e9b806dc0

     기계학습을 위해 이미지 데이터 전처리 및 세트화

     

   - 학습

     CNN 구조를 사용해 AutoEncoder를 구현 

<<<<<<< HEAD
     (LSTM AutoEncoder나 Conv-LSTM AutoEncoder와 같은 구조가 sequence 데이터에 능동적이라 알려져 있으나,

     프레임 사이 한 장의 이미지를 예측하는 상황에서는 CNN 구조만을 사용하는 것이 유리하다고 판단함.)
     
     - Encoder 
     
       Input 이미지열을 Feature벡터로 압축
     
     - Reconstruction Decoder
     
       Feature벡터로 Input 이미지를 재구현
     
     - Prediction Decoder
     
       Feature벡터로 Input 다음 이미지를 예측
     
     <center>Input -> Encoder -> Feature -> Reconstruction Decoder -> Output(1)<center>
     
     <center>Input -> Encoder -> Feature -> Predict Decoder -> Output(2)<center>
     
     Input 이미지를 재구현한 Output(1)과 Input 다음 이미지를 예측한 Output(2)의 오차를 합쳐 BackPropagation
     
     
     
   - 실행(테스트)
   
     테스트 영상을 프레임 단위로 분할, 이미지 데이터 전처리 및 세트화
   
     위의 Output(1)과 Output(2)를 합쳐 프레임 사이 시점의 이미지를 예측해 저장
   
     cv2 모듈을 사용해 이미지 -> 영상
   
     
=======
     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/test.PNG">

     <img src = "https://github.com/durudumba/Fps_augmentation/blob/main/img/result.PNG">


3. **한계**

   프레임 사이 예측이미지에 격자무늬가 생겨 어두운 형태를 띈다. 문제점을 해결하기 위해 정규화 제거, 패딩, 네트워크 변경 등을 시도해봤으나 개선할 수 없었다.
>>>>>>> ce76a6a5a713f73f59eaaee71180be1e9b806dc0

4.  **참조**
   + [LSTMAutoEncoder](https://joungheekim.github.io/2020/10/11/code-review/)
   + [ConvLSTM Encoder-Bidirectional LSTM Decoder](https://hwk0702.github.io/treatise%20review/2021/04/08/ConvLSTAMAE/)
