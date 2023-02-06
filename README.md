# 이미지열 예측을 통한 FPS 증강

1. **개요**

   AutoEncoder를 이용해 영상의 각 프레임 사이에 적합한 이미지를 예측,

   FPS증강을 통해 보다 부드러운 영상출력을 끌어내기 위함이다.

   </br>

2. **과정**

   - 데이터 수집

     **Youtube**에서 키워드 기반으로 URL을 추출해 영상을 다운로드

     

   - 데이터 전처리

     다운받을 영상을 프레임 단위의 이미지로 분할

     기계학습을 위해 이미지 데이터 전처리 및 세팅

     

   - 학습

     CNN 구조를 사용해 AutoEncoder를 구현

     (LSTM AutoEncoder나 ConvLSTM AutoEncoder와 같은 구조가 sequence 데이터에 유연하다고 알려져 있으나,

     프레임 사이 한 장의 이미지를 예측하려는 상황에서 CNN 구조만을 사용하는 것이 적합하다고 판단.)

     - Encoder

       Input 이미지를 Feature 벡터로 압축

     - Reconstrution Decoder

       Feature 벡터로 Input 이미지를 재구현

     - Prediction Decoder

       Feature 벡터로 Input 다음 이미지를 예측

     <center> Input -> Encoder -> Feature -> <b>Reconstruction Decoder</b> -> Output(1) </center>
     <center> Input -> Encoder -> Feature -> <b>Predict Decoder</b> -> Output(2) </center>

     

     Input 이미지를 재구현한 Output(1)과 Input 다음 이미지를 예측한 Output(2)의 오차를 합쳐 Backpropagation

     

   - 실행(테스트)

     테스트 영상을 프레임 단위로 분할, 이미지 데이터 전처리 및 세팅

     위의 Output(1)과 Output(2)를 합쳐 프레임 사이 '시점'의 이미지를 예측해 저장

     cv2 모듈을 사용, 이미지열 -> 영상

     </br>

3. **결과**

   <img src = "./result/concat.png">

   영상에서 표현하는 객체의 위치와 이동에 대한 예측은 가능한 것으로 보임

   그러나, 어둡고 흐릿한 것으로 보아 더 많은 데이터로 많은 학습이 필요한 것으로 예상됨

   </br>

4. **참조**

   - [LSTMAutoEncoder](https://joungheekim.github.io/2020/10/11/code-review/)
   - [ConvLSTM Encoder-Bidirectional LSTM Decoder](https://hwk0702.github.io/treatise%20review/2021/04/08/ConvLSTAMAE/)