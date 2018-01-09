# coding: utf-8
# # Detection with SSD
# In this example, we will load a SSD model and use it to detect objects.
# ### 1. Setup
# * First, Load necessary libs and set up caffe and caffe_root
import sys
sys.path.append('/export/home/xxx/caffe-ssd/python')
import caffe

import numpy as np
import matplotlib.pyplot as plt
#import pprint

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
#import os
print 'set caffe mode...'
caffe.set_mode_gpu()
print 'set caffe devide...'
caffe.set_device(1)

# * Load LabelMap.
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'models/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

model_def = 'models/deploy.prototxt'
model_weights = 'models/person0712_iter_180000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print fmt % (time()-_tstart_stack.pop())


# ### 2. SSD detection

# * Load an image.
# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

thresh=0.2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

# * Load the net in the test phase for inference, and configure input preprocessing.


def det_im(imname):
    image = caffe.io.load_image(imname)
    # * Run the net and examine the top_k results
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    tic()
    detections = net.forward()['detection_out']
    print 'detecting img';toc()
    # Parse the outputs.
    #det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= thresh]

    top_conf = det_conf[top_indices]
    #top_label_indices = det_label[top_indices].tolist()
    #top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    h,w,_=image.shape
    num_p=top_conf.shape[0]
    top_xmin = [max(0,int(round(top_xmin[i] * w))) for i in range(num_p)]
    top_ymin = [max(0,int(round(top_ymin[i] * h))) for i in range(num_p)]
    top_xmax = [min(w,int(round(top_xmax[i] * w))) for i in range(num_p)]
    top_ymax = [min(h,int(round(top_ymax[i] * h))) for i in range(num_p)]
    #pprint.pprint(top_conf)

    #print 'shape:', top_conf.shape, len(top_xmin)
    return top_conf,top_xmin,top_ymin,top_xmax,top_ymax,image

if __name__=='__main__':
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    with open('list_test.txt','r') as f:
        lines=f.readlines()
    for line in lines:
        imname='/export/home/wjh/project_multi/frames_4pc0/'+line.strip('\n')
    # * Plot the boxes
        top_conf,top_xmin,top_ymin,top_xmax,top_ymax,image=det_im(imname)

        plt.imshow(image)
        currentAxis = plt.gca()
        for i in range(top_conf.shape[0]):
            xmin=top_xmin[i]
            ymin=top_ymin[i]
            xmax=top_xmax[i]
            ymax=top_ymax[i]
            score = top_conf[i]
            #label = int(top_label_indices[i])
            label = 1
            #label_name = top_labels[i]
            label_name = 'person'
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

        plt.show()
