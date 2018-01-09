import cv2
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from vis_detect import *
#from vis_detect import tic,toc

import os

def getVisim(im,top_conf,top_xmin,top_ymin,top_xmax,top_ymax):
    im_show=np.copy(im)
    #h,w,_=im.shape

    num_p=top_conf.shape[0]
    for i in range(num_p):
        xmin=top_xmin[i]
        ymin=top_ymin[i]
        xmax=top_xmax[i]
        ymax=top_ymax[i]
        strscore= "%.2f" % top_conf[i]

        cv2.rectangle(im_show,(xmin,ymin),(xmax,ymax),(255,0,0),thickness=2)
        cv2.putText(im_show,strscore,(xmin,ymax-1),cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,0),2)
    return im_show


def getPmatrixFromTxt(txtname):
    with open(txtname, 'r') as f:
        line = f.readline()
        corners = np.array([float(x) for x in line.split()], dtype=np.float32)
        corners = corners.reshape((4, 2))
        line = f.readline()
        rect_corners = np.array([float(x) for x in line.split()], dtype=np.float32)
        rect_corners = rect_corners.reshape((4, 2))
    m = cv2.getPerspectiveTransform(corners, rect_corners)
    m_inverse = cv2.getPerspectiveTransform(rect_corners,corners) 
    return m, m_inverse

def drawpts(im_top, footpts_top):
    for footpt in footpts_top:
        cv2.circle( im_top, (footpt[0],footpt[1]), 10, (0,0,255) , -1)



if __name__=='__main__':

    ### inital matrix
    m_ori2det, m_ori2det_inverse = getPmatrixFromTxt('m_ori2det.txt')
    m_ori2top, m_ori2top_inverse = getPmatrixFromTxt('m_ori2top.txt')
    w_det,h_det = 1280,720
    w_top,h_top = 1280,720

    #imname ='test.jpg'
    datadir='part2'
    imnames= [ datadir+'/' + imname for imname in os.listdir(datadir)]
    imbg=cv2.imread(imnames[0])
    im_top = cv2.warpPerspective(imbg, m_ori2top, (w_top, h_top))


    #for imname in imnames:

    imname = imnames[-2]

    im_ori = cv2.imread(imname)
    im_det = cv2.warpPerspective(im_ori, m_ori2det, (w_det, h_det))
    tmppath='tmp.jpg'
    cv2.imwrite(tmppath, im_det)

    top_conf,top_xmin,top_ymin,top_xmax,top_ymax,_=det_im(tmppath)
    im_show=getVisim(im_det,top_conf,top_xmin,top_ymin,top_xmax,top_ymax)
    cv2.imshow('det', im_show)


    footpts=[[(top_xmin[i]+top_xmax[i])/2.0 , top_ymax[i]] for i in range(top_conf.shape[0])]
    footpts=np.array(footpts,dtype=np.float32)
    footpts=footpts[None,:,:]
    footpts_ori = cv2.perspectiveTransform(footpts, m_ori2det_inverse)
    footpts_top = cv2.perspectiveTransform(footpts_ori, m_ori2top)
    footpts_top = footpts_top[0]

    drawpts(im_top, footpts_top)
    print footpts_top

    cv2.imshow('top', im_top)



    k=cv2.waitKey(0)
    #    if k==27:
    #        cv2.destroyAllWindows()
    #        break
