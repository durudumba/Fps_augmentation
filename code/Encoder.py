import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder,self).__init__()
        self.net = nn.Sequential(                                                       #3,144,144
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3),                     #16,142,142
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),                                                          #16,71,71
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2),           #32,35,35
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.MaxPool2d(2,2)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),           #64,17,17
            nn.ReLU(),
            nn.BatchNorm2d(64),

        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(1, -1)
        return out