import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt
# original name: bbox.py


pnum=18

def getbbox(p0,with_head=False):
    A=np.zeros_like(p0)
    for i in range(pnum):
        if p0[i,2]>0.1:
            A[i,0]=p0[i,0]
            A[i,1]=p0[i,1]

    pl=p0[~np.all(p0==0,axis=1)]

    xmin=pl[:,0].min()
    xmax=pl[:,0].max()
    ymin=pl[:,1].min()
    ymax=pl[:,1].max()

    if with_head==True:
        hadd=(ymax-ymin)*0.15
        #hadd=-30
        print 'hadd:',hadd
        ymin=max(0,(ymin-hadd))



    return [int(xmin),int(ymin),int(xmax),int(ymax)]
    #cv2.circle(im,(int(p0[i,0]),int(p0[i,1])),int(5*p0[i,2]),color[i],2)

def show_frame(ymlfile,imfile,with_head=False):
    data=np.asarray(cv2.cv.Load(ymlfile))
    im=cv2.imread(imfile)
    for i in range(data.shape[0]):

        p0=data[i]
        bbox=getbbox(p0,with_head)
        print bbox
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
    cv2.imshow('test',im)
    cv2.waitKey(0)

def crop_frame(ymlfile,imfilev):
    data=np.asarray(cv2.cv.Load(ymlfile))
    im=cv2.imread(imfile)
    h,w,_=im.shape

    for i in range(data.shape[0]):

        p0=data[i]
        bbox=getbbox(p0)
        print bbox
        r=0.15
        xmin=max(0, bbox[0]-int((bbox[2]-bbox[0])*r))
        xmax=min(w, bbox[2]+int((bbox[2]-bbox[0])*r))
        ymin=max(0, bbox[1]-int((bbox[3]-bbox[1])*r))
        ymax=min(h, bbox[3]+int((bbox[3]-bbox[1])*r))


        cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)
    cv2.imshow('test',im)
    cv2.waitKey(0)

def load_regi(regiyml,regiim):
    data=np.asarray(cv2.cv.Load(regiyml))
    data0=data[0]
    im=cv2.imread(regiim)
    return data0,im

def load_regi_all():
    root='points_labeled'
    with open('regilist.txt','r') as f:
        lines = f.readlines()

    regi_points=np.ndarray((len(lines),18,3))
    for i in range(len(lines)):
        line =lines[i].strip('\n')
        regiyml=root+'/'+line
        data=np.asarray(cv2.cv.Load(regiyml))
        data0=data[0]
        regi_points[i]+=data0
    return regi_points




def get_M(point1,point2):
    row_idx=np.array([1,8,11])
    col_idx=np.array([0,1])
    set1=point1[row_idx[:,None],col_idx]
    set2=point2[row_idx[:,None],col_idx]
    M=cv2.getAffineTransform(set1,set2)
    return M

def do_affine(M):
    [[a,b,tx],[c,d,ty]]=M
    sx=(a**2+b**2)**0.5
    sy=(c**2+d**2)**0.5
    #tan1=b/a
    #tan2=c/d
    print 'M:', M
    if abs(M).sum()==0:
        return False
    elif sx/sy >2 or sy/sx>2:
        print 'sx,xy:',sx,sy
        return False
    else:
        print 'sx,xy:',sx,sy
        return True

def load_yml(ymlfile):
    return np.asarray(cv2.cv.Load(ymlfile))



if __name__=='__main__':

    # yml: key point file from openpose
    ymlroot='../openpose/points/imgs_reID2_sec_1'
    imroot='../openpose/examples/imgs_reID2_sec_1'
    regiyml='points_labeled_sec_1/h1_pose.yml'
    regiim='h1.jpg'
    files=os.listdir(ymlroot)

    p_regi,im_regi=load_regi(regiyml,regiim)
    print im_regi.shape

    for afile in files:
        print afile
        ymlfile=ymlroot+'/'+afile
        imname=afile.split('pose')[0][:-1]+'.jpg'
        imfile=imroot+'/'+imname
        im_buy=cv2.imread(imfile)
        #crop_frame(ymlfile,imfile)
        data=load_yml(ymlfile)
        for i in range(data.shape[0]):
            p_buy=data[i]
            M=get_M(p_buy,p_regi)
            res=cv2.warpAffine(im_buy,M,(im_regi.shape[1],im_regi.shape[0]))
        #cv2.imshow('regi',im_regi)
        #cv2.imshow('res',res)
        #cv2.waitKey(0)
            cv2.imwrite('smallpic/'+imname.split('.')[0]+'_'+str(i)+'.jpg',res)
        #plt.subplot(121)
        #plt.imshow(im_regi)
        #plt.subplot(122)
        #plt.imshow(res)
        #plt.show()
