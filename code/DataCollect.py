from pytube import YouTube
import requests
from bs4 import BeautifulSoup
import re
import cv2
import os

class DataCollect():
    def __init__(self):
        pass

    #검색어로 url추출
    def find_url(keywords):

        urls = []
        for keyword in keywords:    #each keyword : 30urls

            req = requests.get('https://www.youtube.com/results?search_query=' + keyword)
            soup = BeautifulSoup(req.text, 'html.parser')

            scripts = soup.select('script')[-6].get_text()

            pattern = r'[^m]/watch?\S+"'
            matches = re.findall(pattern, scripts)

            for info in matches:
                url_ = info.split(',')[0].replace('"', '')
                urls.append('https://www.youtube.com' + url_)
        print("Extract URL done.")
        return urls

    #영상 다운로드
    def video_download(url, file_name, video_dir):
        yt = YouTube(url)

        videos = yt.streams.filter(file_extension='mp4', res='144p')
        video = videos.order_by('resolution').first()

        down_dir = video_dir

        video.download(output_path=down_dir, filename=file_name)
        print("Download video done.")

    #프레임 단위 분할
    def frame_cut(video_path, image_path):

        if os.path.isfile(video_path):
            vidcap = cv2.VideoCapture(video_path)
        else:
            print("FileNotExist")
            exit(100)
        success, image = vidcap.read()

        count = 0
        success = True

        while success:
            success, image = vidcap.read()
            if success == False :
                break
            
            cv2.imwrite(image_path+str(count)+".jpg", image)

            if (count%10000==0) & (count!=0):
                print("%d.jpg saved." %count)

            # if cv2.waitKey() == 27:
            #     break
            count += 1
        print("Frame_cut done. %d images saved."%count)

        return 0

    def makeImages(keywords, down_dir, img_dir):
        urls = DataCollect.find_url(keywords)
        download_videos = 2

        for i in range(download_videos):
            index = str(i)
            print(f'{index} video task start.')

            DataCollect.video_download(urls[i], (f'{index}.mp4'), down_dir)     ## 영상 다운로드
            os.makedirs(img_dir+index, exist_ok=True)                    ## 프레임 저장위치
            DataCollect.frame_cut(down_dir+index+'.mp4', img_dir+index+'/')     ## 프레임 분할
            print(f'{index} video task done.')



