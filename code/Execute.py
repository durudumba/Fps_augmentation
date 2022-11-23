import os
import av
import cv2

import torch
from glob import glob
import natsort
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from DataCollect import DataCollect
from CustomDataset import CustomDataset


def execute(video, save_video, image_dir, augImg_dir, model, args):
    
    ## Video 4 Image
    os.makedirs(image_dir, exist_ok=True)
    DataCollect.frame_cut(video, image_dir)

    ## Image predict
    img_paths = natsort.natsorted(glob(image_dir+'*.jpg'))

    os.makedirs(augImg_dir, exist_ok=True)
    dataset = CustomDataset(img_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    
    test_iterator = tqdm(enumerate(loader), total=len(loader), desc="Execute")

    model.eval()
    with torch.no_grad():

        for i, batch_item in test_iterator:
            
            _, past_data = batch_item
            batch_size = past_data.size(0)

            past_data = past_data.float().to(args.device)
            outputs = (model.generate(past_data)+model.reconstruct(past_data))/2

            past_data = past_data.reshape(-1, 3, 144, 144).cpu().squeeze(0)
            outputs = outputs.reshape(-1, 3, 144, 144).cpu().squeeze(0)

            for batch in range(batch_size):
                fileN = 2*(batch + (batch_size*i))
                save_image(past_data[batch], augImg_dir+str(fileN)+'.jpg')
                save_image(outputs[batch], augImg_dir+str(fileN+1)+'.jpg')

    # Image 4 Video
    vid = av.open(video).streams.video[0]
    fps = int(str(vid.average_rate).split('/')[0]) / int(str(vid.average_rate).split('/')[1])
    
    h,w,c = cv2.imread(augImg_dir+'0.jpg').shape
    size = (w, h)

    out = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'mp4v'), fps*2, size)

    img_num = tqdm(range(len(glob(augImg_dir+'*.jpg'))), desc="Video Writing")
    for i in img_num:
        img = cv2.imread(augImg_dir+f'{i}.jpg')
        out.write(img)
    out.release()
