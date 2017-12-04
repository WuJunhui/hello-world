#import numpy as np
import cv2
import os

video_name='test_mall.mp4'
dstdir='imgs_small'
if os.path.exists(dstdir):
    os.mkdir(dstdir)
INTERVAL=1


cap = cv2.VideoCapture(video_name)
cnt=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cnt+=INTERVAL
        imname=os.path.join(dstdir, str(cnt).zfill(6)+'.jpg')
        cv2.imwrite(imname,frame)

# Release everything if job is finished
cap.release()
