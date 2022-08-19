from pytube import YouTube
import requests
from bs4 import BeautifulSoup
import re
import cv2
import os
import time

class DataCollect():
    def __init__(self):
        pass

    #검색어로 url추출
    def find_url(keywords):

        urls = []
        for keyword in keywords:    #each keyword : 30urls

            req = requests.get('https://www.youtube.com/results?search_query=' + keyword)
            html = req.text
            soup = BeautifulSoup(html, 'html.parser')

            scripts = soup.select('script')[-6].get_text()

            pattern = r'[^m]/watch?\S+"'
            matches = re.findall(pattern, scripts)

            for info in matches:
                url_ = info.split(',')[0].replace('"', '')
                urls.append('https://www.youtube.com' + url_)
        print("url추출 완료")
        return urls

    #영상 다운로드
    def video_download(url, file_name, video_dir):
        yt = YouTube(url)

        videos = yt.streams.filter(file_extension='mp4', res='144p')
        video = videos.order_by('resolution').first()

        down_dir = video_dir

        video.download(output_path=down_dir, filename=file_name)
        print("영상다운 완료")

    #프레임 단위 분할
    def frame_cut(video_path, image_path):
        if os.path.isfile(video_path):
            vidcap = cv2.VideoCapture(video_path)
        else:
            print("FileNotExist")
            exit()
        success, image = vidcap.read()

        count = 1;
        success = True

        while success:
            success, image = vidcap.read()
            if not success :
                break
            cv2.imwrite(image_path + str(count).zfill(7) + ".jpg", image)
            if(count%1000==0):
                print("saved image %d.jpg" % count)

            if cv2.waitKey(10) == 27:
                break
            count += 1
        print("프레임 단위 분할 완료")

#Train Data Collecting

keywords=['고양이']
down_dir = "../data/video/"
img_dir = "../data/image/"
#video download
urls = DataCollect.find_url(keywords)

#video cut by frame
for i in range(15):
    index = str(i+1)

    start = time.time()
    DataCollect.video_download(urls[i], index + ".mp4", down_dir)
    print("%s 번 째 영상작업 중..."%index)

    os.makedirs(img_dir+index.zfill(2), exist_ok=True)
    DataCollect.frame_cut(down_dir+index+".mp4", img_dir+index+"/")

    runtime = time.time()-start
    print("%s 번 째 영상 작업시간 : %d sec" %(index, runtime))
    print("%s 번 째 영상 작업완료" %index)

# 실험영상 프레임 컷
image_path = '../test/cat video.mp4'
DataCollect.frame_cut('../cat_video.mp4', '../cat_video/')
