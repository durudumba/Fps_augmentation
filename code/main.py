#Library
import glob
from PIL import Image
import shutil
import os
import cv2
import av

from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch

import DataCollect

#base
video_name='beach_video'
video = '../test/%s.mp4'%video_name
dataset_folder = '../test/%s/'%video_name
os.makedirs(dataset_folder, exist_ok=True)
destination_folder = '../test/%s_prediction/'%video_name
os.makedirs(destination_folder, exist_ok=True)

# Video to Image
os.makedirs(dataset_folder+'새 폴더/', exist_ok=True)
DataCollect.frame_cut(video, dataset_folder+'새 폴더/')

# Image data preprocess
transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor()])
dataset = datasets.ImageFolder(dataset_folder, transform=transform)

# GPU활성화 및 encoder,decoder load
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder, decoder = torch.load('../model/autoencoder_SeqImgPred.pkl')

# Image sequence prediction and Save image
for index in range(dataset.__len__()-2):

    before = dataset.__getitem__(index)[0].unsqueeze(0).to(device)
    after = dataset.__getitem__(index+1)[0].unsqueeze(0).to(device)

    before_fmap = encoder(before)
    after_fmap = encoder(after)
    #Feature map 합성
    fmap = ((before_fmap+after_fmap)/2)
    output = decoder(fmap).squeeze(0)

    save_image(output, '%s%sp.jpg'%(destination_folder,str(index+1).zfill(7)))

# Image resize //(144,144) -> (256,144)
names=os.listdir(destination_folder)
for name in names:
    path = destination_folder+name
    img = Image.open(path)
    ir = img.resize((256,144))
    ir.save(path)

# Original image+prediction image
orignal_path = dataset_folder+'새 폴더/'
orignals=os.listdir(orignal_path)
for orignal in orignals:
    path = orignal_path+orignal
    shutil.copy(path, destination_folder)

# Images to video
# 참고자료 : https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
whole = os.listdir(destination_folder)
vid = av.open(video).streams.video[0]
fps = int(str(vid.average_rate).split('/')[0]) / int(str(vid.average_rate).split('/')[1])

img_array=[]
for filename in glob.glob(destination_folder+'*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('../test/%s_worked.avi'%video_name,cv2.VideoWriter_fourcc(*'DIVX'),fps*2, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
