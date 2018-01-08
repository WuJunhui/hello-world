import os
#import sys
import numpy as np
import cv2

import shutil
import _init_paths

import caffe

# use cv2, one channel

class CaffeForward:

    def __init__(self, net_def, weights, mean_file):
        self.im_shape=224
        self.net_ = caffe.Net(net_def, weights, caffe.TEST)
        n_channels, height, width = [x for x in self.net_.blobs['data'].shape][-3:]
        self.net_.blobs['data'].reshape(1, n_channels, height, width)

        if mean_file.endswith('npy'):
            self.mean = np.load(mean_file)[0].mean(1).mean(1)
            #print(channels_mean, channels_mean.dtype, type(channels_mean))
        else:
            with open(mean_file, 'r') as f:
                self.mean = np.array([float(x) for x in f])
                #print(channels_mean, channels_mean.dtype, type(channels_mean))

    def forward(self, img_path):
        im = cv2.imread(img_path)
        im = cv2.resize( im, (self.im_shape, self.im_shape), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32, copy=False)
        im -= self.mean
        im = im.transpose((2,0,1))

        self.net_.blobs['data'].data[...] = im
        self.net_.forward()

    def get_blob(self, blob_name):
        return self.net_.blobs[blob_name].data

    def get_img_feature(self, img_path, blob_name):
        self.forward(img_path)
        return self.get_feature(blob_name)

    def get_params(self, layer_name):
        return self.net_.params[layer_name]

#if __name__ == '__main__':

    #if len(sys.argv) < 5:
    #    print('Use: python cnn_fe.py [model_file] [model_weights] [model_mean] [image_list]')
    #    sys.exit(0)

    #model_file = sys.argv[1]
    #model_weights = sys.argv[2]
    #model_mean = sys.argv[3]
    #img_path = sys.argv[4]
    #gpu_id = int(sys.argv[5])

def demo(model_weights):
    #write a data layer to txt
    model_file = 'deploy-cls2.prototxt'
    #model_weights = 'snapshot/train_iter_124000.caffemodel'
    #model_weights = sys.argv[1]
    model_mean = 'meanax.txt'
    root=''
    img_list = root+'/'+'valall.txt'
    gpu_id = 2
    dataroot=root+'/'+'val_v2'
    dstf=open('outfc/outfc_{}.txt'.format(model_weights.split('.')[0].split('_')[-1]),'w')
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    cf = CaffeForward(model_file, model_weights, model_mean)

    with open(img_list, 'r') as f:
        line = f.readline().rstrip()
        while line:
            img_path = dataroot+'/'+line.split()[0]
            cf.forward(img_path)
            probs = cf.get_blob('fc_2_class')
            outline = ' '.join([line]+[str(x) for x in probs[0]])
            dstf.write(outline+'\n')
            line = f.readline().rstrip()


if __name__ == '__main__':
    weightsdir='snapshot'
    model_weights=[os.path.join(weightsdir,f) for f in os.listdir(weightsdir) if f.split('.')[-1]=='caffemodel']
    model_weights=model_weights[-3:]
    print model_weights
    raw_input()
    for model_weight in model_weights:
        demo(model_weight)
        shutil.move(model_weight, 'snapshot_0')
        shutil.move(model_weight.split('.')[0]+'.'+'solverstate', 'snapshot_0')
