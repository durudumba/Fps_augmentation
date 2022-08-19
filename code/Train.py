#Library
from Encoder import Encoder
from Decoder import Decoder

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset_folder = '../data/image/'
    # Hyperparameters 설정
    learning_rate = 0.0001
    num_epochs = 3

    # Image data preprocessing
    transform = transforms.Compose([
        transforms.Resize((144,144)),
        transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_folder, transform=transform)

    # Set GPU, Parameter, LossFunction, Optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # Training
    loss_record=[]
    for i in range(num_epochs):
        start=time.time()
        for index in range(dataset.__len__()-2):

            optimizer.zero_grad()
            before = dataset.__getitem__(index)[0].unsqueeze(0).to(device)
            target = dataset.__getitem__(index+1)[0].to(device)
            after = dataset.__getitem__(index+2)[0].unsqueeze(0).to(device)

            before_feature = encoder(before)
            after_feature = encoder(after)
            # Feature map 합성
            xfmap = ((before_feature+after_feature)/2)
            output = decoder(xfmap).squeeze(0)

            # Loss Calculate
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

            if (index % 1000 == 0):
                print("%d image processing : " % index, end='')
                print(loss)

        print("\nepochs : {0}, loss : {1}".format(i, loss))
        used = time.time()-start
        print("{0} min {1} sec used\n".format(int(used/60), int(used%60)))
        torch.save([encoder, decoder], '../model/autoencoder_SeqImgPred.pkl')

    # Loss 곡선그래프
    plt.plot(loss_record)
    plt.show()




