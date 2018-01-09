import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vis_detect import *
from vis_detect import tic,toc

#import os
cnt_match=0
cnt_alone=0

NUM=2
H=360
W=360
M=np.array([
[[0.176138, 0.647589, -63.412272],
 [-0.180912, 0.622446, -0.125533],
 [-0.000002, 0.001756, 0.10231]],
[[0.177291,0.004724, 31.224545],
 [  0.169895,0.661935,-79.781865],
 [ -0.000028,0.001888,  0.054634]],
[[-0.118791,	0.077787,	64.819189],
 [0.133127	,0.069884	,15.832922],
 [-0.000001,	0.002045,	-0.057759]],
[[-0.142865,	0.553150,	-17.395045],
 [-0.125726,	0.039770,	75.937144],
 [-0.000011,	0.001780,	0.015675]]])

def getImpath(frameid):
    impaths=[]
    for camid in range(NUM):
        imname='frames_4pc'+str(camid)+'/'+str(frameid).zfill(6)+'.jpg'
        impaths.append(imname)
    return impaths

def getWarpImsfromIms(ims,camid):
    warpims=[]
    for im_in in ims:
        #cv2.imshow('imin'+str(camid),im_in)
        res=cv2.warpPerspective(im_in,M[camid],(W,H))
        warpims.append(res)
        #cv2.imshow('res'+str(camid),res)
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.destroyWindow('imin')
    return warpims

def getWarpImsFrameid(frame_count=0):
    impaths=getImpath(frame_count)
    res=[]
    for i in range(NUM):
        im=cv2.imread(impaths[i])
        ares=cv2.warpPerspective(im,M[i],(W,H))
        res.append(ares)
    resMer=cv2.addWeighted(res[0],0.5,res[1],0.5,0)
    return resMer


def getMask(impath):
    im=cv2.imread(impath)
    top_conf,top_xmin,top_ymin,top_xmax,top_ymax,_=det_im(impath)
    num_p=top_conf.shape[0]
    im_masks=[]
    detbboxs=[]
    im_peoples=[]
    for i in range(num_p):
        im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_mask=np.zeros_like(im_gray)
        xmin=top_xmin[i]
        ymin=top_ymin[i]
        xmax=top_xmax[i]
        ymax=top_ymax[i]
        im_people=im[ymin:ymax,xmin:xmax,:]
        cv2.rectangle(im_mask,(xmin,ymin),(xmax,ymax),(255,255,255),thickness=-1)
        im_masks.append(im_mask)
        detbboxs.append([xmin,ymin,xmax,ymax])
        im_peoples.append(im_people)
    return im_masks,detbboxs,im_peoples


def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # bboxA[xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0.0,(xB - xA + 1)) * max(0.0,(yB - yA + 1))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def getRatio(maskA,rectA,maskwA,camidA,maskB,rectB,maskwB,camidB):
    im_added=cv2.addWeighted(maskwA,0.5, maskwB,0.5,0)
    _,im_cross =cv2.threshold(im_added,200,255,cv2.THRESH_BINARY)
    carea=(im_cross-np.zeros_like(im_added)).sum()
    carea=carea/255
    #if not (im_cross- np.zeros_like(im_added)).any():
    if carea==0:
        print 'no cross area find.'
        return 0,0,0
    masks=[maskA,maskB]
    rects=[rectA,rectB]
    camids=[camidA,camidB]
    rdowns=[]
    for i in range(2):
        inverse=cv2.warpPerspective(im_cross,M[camids[i]],(masks[i].shape[1],masks[i].shape[0]),flags=cv2.WARP_INVERSE_MAP)
        #cv2.imshow('inverse1',inverse)
        contours, hierarchy = cv2.findContours(inverse,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            print 'len contours', len(contours)
            print 'no contours area find.'
            return 0,0,0
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)

        rdown=(y+h/2.0-rects[i][1])/float(rects[i][3]-rects[i][1])
        rdowns.append(rdown)

        #cv2.rectangle(inverse,(x,y),(x+w,y+h),(255,255,255),2)
    print 'matching ratio:',rdowns[0],rdowns[1],carea
    return rdowns[0],rdowns[1],carea

def getVisim(im,rects):
    im_copy=np.copy(im)
    h,w,_=im.shape
    #r=0.3
    for rectid in range(len(rects)):
        rect=rects[rectid]
        cv2.rectangle(im_copy,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),2)
        cv2.putText(im_copy,str(rectid),(rect[0],rect[3]),cv2.FONT_HERSHEY_COMPLEX,2, (0,0,255),2)
    #im_s=cv2.resize(im_copy,(int(w*r),int(h*r)))
    #print 'im_s.shape',im_s.shape
    return im_copy




def demoShowMatch(frameid):
    impaths = getImpath(frameid)
    all_masks=[]
    all_masks_warp=[]
    all_detbboxs=[]
    all_peoples=[]
    for i in range(NUM):
        im_masks,detbboxs,im_peoples=getMask(impaths[i])
        im_masks_warp=getWarpImsfromIms(im_masks,i)
        all_masks.append(im_masks)
        all_masks_warp.append(im_masks_warp)
        all_detbboxs.append(detbboxs)
        all_peoples.append(im_peoples)

    camidA=0
    camidB=1

    pnumA=len(all_masks[camidA])
    print 'pnumA:', pnumA
    pnumB=len(all_masks[camidB])
    print 'pnumB:', pnumB

    tic()
    imA=cv2.imread(impaths[camidA])
    imB=cv2.imread(impaths[camidB])
    imA=getVisim(imA,all_detbboxs[camidA])
    imB=getVisim(imB,all_detbboxs[camidB])
    print 'cv2draw time:'
    toc()

    tic()
    fig0=plt.figure(0,figsize=(6,10))

    plt.subplot(211)
    plt.title('imA')
    plt.imshow(cv2.cvtColor(imA, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(cv2.cvtColor(imB, cv2.COLOR_BGR2RGB))
    plt.title('imB')
    plt.axis('off')

    plt.savefig('fig0.jpg')
    fig0.clear()
    print 'pltplot0 time:'
    toc()


    #cal matrix between two image
    matAB=np.zeros((pnumA,pnumB,3),dtype=float)
    #cross in A, corss in B, iou
    for idA in range(pnumA):
        maskA=all_masks[camidA][idA]
        rectA=all_detbboxs[camidA][idA]
        maskwA=all_masks_warp[camidA][idA]
        for idB in range(pnumB):
            maskB=all_masks[camidB][idB]
            rectB=all_detbboxs[camidB][idB]
            maskwB=all_masks_warp[camidB][idB]

            print 'matching idA: %d idB: %d ...' %(idA,idB)
            matAB[idA,idB,:]=np.array(getRatio(maskA,rectA,maskwA,camidA,maskB,rectB,maskwB,camidB))

    print 'matAB.shape:', matAB.shape
    print 'matAB:'
    print matAB

    th_cross=0.7
    #cal true or false
    matABm=np.zeros((pnumA,pnumB),dtype=float)
    for idA in range(pnumA):
        for idB in range(pnumB):
            ratio=(max(0,matAB[idA,idB,0]-th_cross))*max(0,(matAB[idA,idB,1]-th_cross))*matAB[idA,idB,0]
            matABm[idA,idB]=ratio

    print 'matABm:'
    print matABm
    matAB2=matABm.astype(bool)
    for idA in range(pnumA):
        if matAB2[idA,:].sum()>1:
            maxid=matABm[idA,:].argmax()
            matAB2[idA,:]=False
            matAB2[idA,maxid]=True
    for idB in range(pnumB):
        if matAB2[:,idB].sum()>1:
            maxid=matABm[:,idB].argmax()
            matAB2[:,idB]=False
            matAB2[maxid,idB]=True

    print 'matAB2:'
    print matAB2
    tic()
    #show people matche and alone
    subcnt=0
    fig1=plt.figure(1,figsize=(10,10))
    plt.axis('off')
    for idA in range(pnumA):
        for idB in range(pnumB):
            if matAB2[idA][idB]==True:
                print 'p in camA id %d and camB id %d match' %(idA,idB)
                impA=np.copy(all_peoples[camidA][idA])
                impB=np.copy(all_peoples[camidB][idB])

                fig1.add_subplot(4,4,subcnt*4+1)
                plt.imshow(cv2.cvtColor(impA, cv2.COLOR_BGR2RGB))
                plt.title('A'+str(idA))
                plt.axis('off')

                fig1.add_subplot(4,4,subcnt*4+2)
                plt.imshow(cv2.cvtColor(impB, cv2.COLOR_BGR2RGB))
                plt.title('B'+str(idB))
                plt.axis('off')
                subcnt+=1
                global cnt_match
                cnt_match+=1

    #show people alone
    subcnt=0
    for idA in range(pnumA):
        if not matAB2[idA,:].any():
            print 'p in camA id %d is alone' %idA
            impA=all_peoples[camidA][idA]
            fig1.add_subplot(4,4,subcnt*4+3)
            plt.imshow(cv2.cvtColor(impA, cv2.COLOR_BGR2RGB))
            plt.title('A'+str(idA))
            plt.axis('off')
            subcnt+=1
            global cnt_alone
            cnt_alone+=1

    subcnt=0
    for idB in range(pnumB):
        if not matAB2[:,idB].any():
            print 'p in camB id %d is alone' %idB
            impB=all_peoples[camidB][idB]
            fig1.add_subplot(4,4,subcnt*4+4)
            plt.imshow(cv2.cvtColor(impB, cv2.COLOR_BGR2RGB))
            plt.title('B'+str(idB))
            plt.axis('off')
            subcnt+=1
            global cnt_alone
            cnt_alone+=1
    plt.savefig('fig1.jpg')
    print 'pltfig1 time:'
    fig1.clear()
    toc()

    tic()
    imfig0=cv2.imread('fig0.jpg')
    imfig1=cv2.imread('fig1.jpg')
    cv2.imshow('ims',imfig0)
    cv2.imshow('match and alone',imfig1)
    cv2.waitKey(0)
    print 'show time:'
    toc()

def getFootpts(detbboxs,m):
    footpts=[]
    for detbbox in detbboxs:
        bottom_x = (detbbox[0]+detbbox[2]) / 2.0
        bottom_y = detbbox[3]

        src=np.array([[bottom_x,bottom_y]],dtype=np.float32)
        src=src[None,:,:]
        res=cv2.perspectiveTransform(src,m)
        footpts.append((res[0][0][0],res[0][0][1]))
        #projected = np.dot(m, [bottom_x, bottom_y, 1])
        #proj_x = int(projected[0] / projected[2] + 0.5)
        #proj_y = int(projected[1] / projected[2] + 0.5)
        #footpts.append((proj_x,proj_y))
    print 'footpts', footpts
    return footpts



def demoDrawTop(frameid):
    impaths = getImpath(frameid)
    all_masks=[]
    all_masks_warp=[]
    all_detbboxs=[]
    all_peoples=[]
    for i in range(NUM):
        im_masks,detbboxs,im_peoples=getMask(impaths[i])
        im_masks_warp=getWarpImsfromIms(im_masks,i)
        all_masks.append(im_masks)
        all_masks_warp.append(im_masks_warp)
        all_detbboxs.append(detbboxs)
        all_peoples.append(im_peoples)

    all_footpts=[getFootpts(all_detbboxs[i],M[i]) for i in range(NUM)]

    camidA=0
    camidB=1

    pnumA=len(all_masks[camidA])
    print 'pnumA:', pnumA
    pnumB=len(all_masks[camidB])
    print 'pnumB:', pnumB

    tic()
    imA=cv2.imread(impaths[camidA])
    imB=cv2.imread(impaths[camidB])
    imA=getVisim(imA,all_detbboxs[camidA])
    imB=getVisim(imB,all_detbboxs[camidB])
    print 'cv2draw time:'
    toc()

    tic()
    fig0=plt.figure(0,figsize=(6,10))

    plt.subplot(211)
    plt.title('imA')
    plt.imshow(cv2.cvtColor(imA, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(cv2.cvtColor(imB, cv2.COLOR_BGR2RGB))
    plt.title('imB')
    plt.axis('off')

    plt.savefig('fig0.jpg')
    fig0.clear()
    print 'pltplot0 time:'
    toc()


    #cal matrix between two image
    matAB=np.zeros((pnumA,pnumB,3),dtype=float)
    #cross in A, corss in B, iou
    for idA in range(pnumA):
        maskA=all_masks[camidA][idA]
        rectA=all_detbboxs[camidA][idA]
        maskwA=all_masks_warp[camidA][idA]
        for idB in range(pnumB):
            maskB=all_masks[camidB][idB]
            rectB=all_detbboxs[camidB][idB]
            maskwB=all_masks_warp[camidB][idB]

            print 'matching idA: %d idB: %d ...' %(idA,idB)
            matAB[idA,idB,:]=np.array(getRatio(maskA,rectA,maskwA,camidA,maskB,rectB,maskwB,camidB))

    print 'matAB.shape:', matAB.shape
    print 'matAB:'
    print matAB

    th_cross=0.7
    #cal true or false
    matABm=np.zeros((pnumA,pnumB),dtype=float)
    for idA in range(pnumA):
        for idB in range(pnumB):
            ratio=(max(0,matAB[idA,idB,0]-th_cross))*max(0,(matAB[idA,idB,1]-th_cross))*matAB[idA,idB,2]
            matABm[idA,idB]=ratio

    print 'matABm:'
    print matABm
    matAB2=matABm.astype(bool)
    for idA in range(pnumA):
        if matAB2[idA,:].sum()>1:
            maxid=matABm[idA,:].argmax()
            matAB2[idA,:]=False
            matAB2[idA,maxid]=True
    for idB in range(pnumB):
        if matAB2[:,idB].sum()>1:
            maxid=matABm[:,idB].argmax()
            matAB2[:,idB]=False
            matAB2[maxid,idB]=True

    print 'matAB2:'
    print matAB2
    tic()
    #show people matche and alone
    subcnt=0
    fig1=plt.figure(1,figsize=(10,10))
    plt.axis('off')


    topbg=getWarpImsFrameid(frameid)
    for idA in range(pnumA):
        for idB in range(pnumB):
            if matAB2[idA][idB]==True:
                print 'p in camA id %d and camB id %d match' %(idA,idB)
                impA=np.copy(all_peoples[camidA][idA])
                impB=np.copy(all_peoples[camidB][idB])

                footptA=all_footpts[camidA][idA]
                footptB=all_footpts[camidB][idB]
                footpt= (int((footptA[0]+footptB[0]) /2.0 +0.5) , int((footptA[1]+footptB[1]) /2.0 +0.5))
                cv2.circle(topbg,footpt,3,(0,0,255),thickness=-1)



                fig1.add_subplot(4,4,subcnt*4+1)
                plt.imshow(cv2.cvtColor(impA, cv2.COLOR_BGR2RGB))
                plt.title('A'+str(idA))
                plt.axis('off')

                fig1.add_subplot(4,4,subcnt*4+2)
                plt.imshow(cv2.cvtColor(impB, cv2.COLOR_BGR2RGB))
                plt.title('B'+str(idB))
                plt.axis('off')
                subcnt+=1
                global cnt_match
                cnt_match+=1

    #show people alone
    subcnt=0
    for idA in range(pnumA):
        if not matAB2[idA,:].any():
            print 'p in camA id %d is alone' %idA
            impA=all_peoples[camidA][idA]

            footptA=all_footpts[camidA][idA]
            cv2.circle(topbg,footptA,3,(0,0,255),thickness=-1)


            fig1.add_subplot(4,4,subcnt*4+3)
            plt.imshow(cv2.cvtColor(impA, cv2.COLOR_BGR2RGB))
            plt.title('A'+str(idA))
            plt.axis('off')
            subcnt+=1
            global cnt_alone
            cnt_alone+=1

    subcnt=0
    for idB in range(pnumB):
        if not matAB2[:,idB].any():
            print 'p in camB id %d is alone' %idB
            impB=all_peoples[camidB][idB]

            footptB=all_footpts[camidB][idB]
            cv2.circle(topbg,footptB,3,(0,0,255),thickness=-1)


            fig1.add_subplot(4,4,subcnt*4+4)
            plt.imshow(cv2.cvtColor(impB, cv2.COLOR_BGR2RGB))
            plt.title('B'+str(idB))
            plt.axis('off')
            subcnt+=1
            global cnt_alone
            cnt_alone+=1
    plt.savefig('fig1.jpg')
    print 'pltfig1 time:'
    fig1.clear()
    toc()

    tic()
    imfig0=cv2.imread('fig0.jpg')
    imfig1=cv2.imread('fig1.jpg')
    cv2.imshow('ims',imfig0)
    cv2.imshow('topbg',topbg)
    cv2.imshow('match and alone',imfig1)
    print 'show time:'
    toc()



if __name__=='__main__':

    frameid=1200
    while frameid<3914:
        print 'processing frameid %d ......' %frameid
        demoDrawTop(frameid)
        k=cv2.waitKey(0)
        if k==27:
            break

        frameid+=1
    print 'cntmatch,alone:',cnt_match,cnt_alone
    print 'all done!'


