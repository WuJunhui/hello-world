import cv2
import numpy as np

NUM=4
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

def getWarpIms(frameid):
    warpims=[]
    for camid in range(NUM):
        imname='frames_4pc'+str(camid)+'/'+str(frameid).zfill(6)+'.jpg'
        im_in=cv2.imread(imname)
        cv2.imshow('imin',im_in)
        res=cv2.warpPerspective(im_in,M[camid],(W,H))
        warpims.append(res)
        #cv2.imwrite(str(camid)+'.jpg',res)
        cv2.imshow('res'+str(camid),res)
        cv2.waitKey(100)
    #cv2.destroyAllWindows()
    cv2.destroyWindow('imin')
    return warpims

def getBlurIms(ims):
    blurims=[]
    for i in range(NUM):
        bl = cv2.cvtColor(ims[i],cv2.COLOR_BGR2GRAY)
        bl = cv2.GaussianBlur(bl,(21,21),0)
        cv2.imshow('bl',bl)
        cv2.waitKey(100)
        blurims.append(bl)
    #cv2.destroyAllWindows()
    cv2.destroyWindow('bl')
    return blurims
def getDiff(grey_bgs,grey_frames):
    diffs=[]
    for i in range(NUM):
        diff=cv2.absdiff(grey_bgs[i],grey_frames[i])
        _,diff=cv2.threshold(diff,25,63,cv2.THRESH_BINARY)
        cv2.imshow('diff',diff)
        cv2.waitKey(100)
        diffs.append(diff)
    cv2.destroyWindow('diff')
    return diffs


if __name__=='__main__':
    frameid=360 
    #frameid=690
    #frameid=1200
    bgs=getWarpIms(0)
    grey_bgs=getBlurIms(bgs)

    frames=getWarpIms(frameid)
    grey_frames=getBlurIms(frames)
    diffs=getDiff(grey_bgs,grey_frames)
    
    diffall=np.zeros_like(diffs[0],dtype=np.uint8)
    for i in range(NUM):
        diffall+=diffs[i]
    
    #_,diffall=cv2.threshold(diffall,64,255,cv2.THRESH_TOZERO)

    cv2.imshow('diffall',diffall)
    print diffall.max()
    print diffall.argmax()
    cv2.waitKey(0)
    print 'Done!'
