import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder,self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=3,stride=2,output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        out = x.view(1,64,17,17)
        out = self.net(out)
        return out