import os
import sys
sys.path.append('/export/home/wjh/RealPose/caffe_train/python')
import caffe
import numpy as np
import cv2
import bbox
import cnn_fe
# use openpose key points to detect people, then affine transform, then reid.
# yml file: form openpose

def find_ID(vec,val_vecs,impp):
    num_val=val_vecs.shape[0]
    dists=np.zeros([num_val,])
    for i in range(num_val):
        dist=((val_vecs[i,:]-vec)**2).sum()
        dists[i]+=dist
    val_id=dists.argmin()
    person_id=val_id/impp
    return val_id,person_id

if __name__ == '__main__':

    model_file = 'jstl_dgd_deploy_inference.prototxt'
    model_weights = 'jstl_dgd_inference.caffemodel'
    model_mean = 'mean.txt'
    caffe.set_device(0)
    caffe.set_mode_gpu()
    cf = cnn_fe.CaffeForward(model_file, model_weights, model_mean)

    val_vecs=np.load('regi_vecs_sec1.npy')
    impp=3

    names=['ZAX','CJY','CGF','HZB']
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
    imroot='../openpose/examples/imgs_reID2_sec_1'
    ymlroot='../openpose/points/imgs_reID2_sec_1'

    labeled_imroot='labeled_sec_1'
    #'labeled_data_affine'

    #regiyml='points_labeled/AnS/0_2_pose.yml'
    #regiim='labeled_data/AnS/0_2.jpg'
    regiyml='h1_pose.yml'
    regiim='h1.jpg'
    p_regi,im_regi=bbox.load_regi(regiyml,regiim)

    #with open('regilist.txt','r') as f:
    with open('regilist_sec1.txt','r') as f:
        regilines=f.readlines()


    files = os.listdir(ymlroot)
    for afile in files:
        ymlfile=ymlroot+'/'+afile
        imname=afile.split('_')[0]+'.jpg'
        imfile=imroot+'/'+imname
        im_buy=cv2.imread(imfile)
        data=bbox.load_yml(ymlfile)

        for i in range(data.shape[0]):
            p_buy=data[i]
            [xmin,ymin,xmax,ymax] = bbox.getbbox(p_buy,True)

            M=bbox.get_M(p_buy,p_regi)
            if bbox.do_affine(M):
                res=cv2.warpAffine(im_buy,M,(im_regi.shape[1],im_regi.shape[0]))
            else:
                res=im_buy[ymin:ymax,xmin:xmax,:]
            cv2.imwrite('tmp.jpg',res)
            #raw_input()

            img_path_test='tmp.jpg'
            cf.forward(img_path_test)
            feats = cf.get_blob('fc7')

            regi_id,person_id=find_ID(feats,val_vecs,impp)
            print 'regi_id',regi_id
            regi_im= cv2.imread(labeled_imroot+'/'+regilines[regi_id].strip('\n'))

            im_buy_toshow=np.zeros_like(im_buy)
            im_buy_toshow+=im_buy

            cv2.rectangle(im_buy_toshow,(xmin,ymin),(xmax,ymax),colors[person_id],2)
            cv2.putText(im_buy_toshow,names[person_id],(xmin,ymin),cv2.FONT_HERSHEY_COMPLEX, 2, colors[person_id])

            cv2.imshow('img',im_buy_toshow)
            cv2.imshow('regi_im',regi_im)
            cv2.imshow('res_im',res)

            cv2.waitKey(2000)
