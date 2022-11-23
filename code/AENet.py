import torch.nn as nn
from torchvision.models import efficientnet_b0
from Encoder import Encoder
from Decoder import Decoder

class AENet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder().to(args.device)
        self.predict_decoder = Decoder().to(args.device)
        self.reconstruction_decoder = Decoder().to(args.device)
        self.criterion = nn.MSELoss().to(args.device)

    # Loss 연산
    def forward(self, input, target):

        ## Encoder
        encoder_output = self.encoder(input)

        ## Prediction Loss
        feature = encoder_output
        predict_output = self.predict_decoder(feature)
        predict_loss = self.criterion(predict_output, target)

        ## Reconstruction Loss
        feature = encoder_output
        reconstruction_output = self.reconstruction_decoder(feature)
        reconstruction_loss = self.criterion(reconstruction_output, input)

        loss = (reconstruction_loss + predict_loss)/2

        return loss

    
    # 이미지 생성
    def generate(self, input):
        
        ## Encoder
        encoder_output = self.encoder(input)
        
        feature = encoder_output
        predict_output = self.predict_decoder(feature)

        return predict_output

    # 이미지 복구
    def reconstruct(self, input):

        ## Encoder
        encoder_output = self.encoder(input)

        feature = encoder_output
        reconstruction_output = self.reconstruction_decoder(feature)

        return reconstruction_output
        
    
