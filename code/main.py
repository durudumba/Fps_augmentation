from DataCollect import DataCollect
from Train import train
from CustomDataset import CustomDataset
from Execute import execute
from AENet import AENet

import easydict
from glob import glob
import os
import natsort

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# 영상 수집 및 이미지화
keywords = ['고양이']       ## 키워드 당 15개의 영상을 검색
down_dir = './data/video/'   ## 다운받을 영상위치
img_dir = './data/image/'    ## 프레임 이미지 저장 위치
DataCollect.makeImages(keywords, down_dir, img_dir)


# 하이퍼파라미터 설정
args = easydict.EasyDict({
    "batch_size": 128, 
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "learning_rate" : 5e-5,
    "epochs" : 20,
})

# 학습 데이터 구축
img_path = natsort.natsorted(glob('./data/image/*/*.jpg'))
train_path, val_path= train_test_split(img_path, test_size=0.2)

train_dataset = CustomDataset(train_path)
val_dataset = CustomDataset(val_path)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)


# 모델
mode = 'train'              ## 'train' : 모델 학습진행 / 'load' : 모델 로드진행
model = AENet(args)
model_name = 'model.bin'    ## 학습/로드 할 모델명
model_dir = './result/'     ## 모델 저장/로드 위치
os.makedirs(model_dir, exist_ok=True)

if mode=='train':

    ## 모델 학습
    model.to(args.device)

    model = train(args, model, train_dataloader, val_dataloader)

    ## 모델 저장
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir+model_name)

elif mode=='load':
    ## 모델 호출
    model.load_state_dict(torch.load(model_dir+model_name))
    model.to(args.device)

# 테스트

test_video = './result/catvideo.mp4'      ## 증강할 영상 위치
save_video = './result/catvideo_augmented.mp4'    ## 증강한 영상 저장 위치
test_image_dir = './result/images/'        ## 프레임 컷 할 위치
test_AugImg_dir = './result/images_aug/'   ## 이미지 증강 저장할 위치

execute(test_video, save_video, test_image_dir, test_AugImg_dir, model, args)


