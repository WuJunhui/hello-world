import caffe

import numpy as np
import cv2
import scipy.misc
from random import shuffle
#
# 1. forward a batch for seg data
# 2. use cv2 to load and resize data
# 3. seg and cls task together

class YELLOWContextSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL-Context
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 59 class task defined by

        R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R.
        Urtasun, and A. Yuille.  The Role of Context for Object Detection and
        Semantic Segmentation in the Wild.  CVPR 2014.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC dir (must contain 2010)
        - context_dir: path to PASCAL-Context annotations
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL-Context semantic segmentation.

        example: params = dict(voc_dir="/path/to/PASCAL", split="val")
        """
        # config
        params = eval(self.param_str)

        check_params(params)
        self.batch_size = params['batch_size']
        self.iter_cls = params['iter_cls']
        self.iter_cur = 0
        self.batch_loader = BatchLoader(params, None)

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data and 2 label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indexlist_seg for images and labels

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        top[0].reshape(self.batch_size, 3, params['im_shape'], params['im_shape'])
        # Note the 20 channels (because PASCAL has 20 classes.)
        #top[1]: seg label
        top[1].reshape(self.batch_size, 1, params['im_shape'],params['im_shape'] )
        #top[2]: cls label
        top[2].reshape(self.batch_size, 1,1,1)


    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass


    def forward(self, bottom, top):

        if self.iter_cur != self.iter_cls:
            self.task_cls=True
            self.iter_cur +=1
        else:
            self.task_cls=False
            self.iter_cur =0

        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label_seg, label_cls = self.batch_loader.load_next_sample(self.task_cls)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label_seg
            top[2].data[itt, ...] = label_cls

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    """
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.split = params['split']
        self.mean = np.array((104.007, 116.669, 122.679), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.im_shape = params.get('im_shape', 224)

        # for seg
        self.voc_dir = params['voc_dir']
        self.context_dir = params['context_dir']
        self.seg_cur = 0  # current image
        split_f_seg  = '{}/ImageSets/Main/{}.txt'.format(self.voc_dir,self.split)
        self.indexlist_seg = open(split_f_seg, 'r').read().splitlines()
        if params['split']=='train':
            shuffle(self.indexlist_seg)

        #for cls
        self.cls_dir = params['cls_dir']
        self.cls_cur = 0  # current image
        split_f_cls  = '{}/{}.txt'.format(self.cls_dir,self.split)
        self.indexlist_cls = open(split_f_cls, 'r').read().splitlines()
        if params['split']=='train':
            shuffle(self.indexlist_cls)


    def load_next_sample(self,task_cls=True):
        if task_cls:
            im, label_cls = self.load_next_sample_cls()
            label_seg = np.ones((1,self.im_shape,self.im_shape),np.uint8) * 255

        else:
            im, label_seg = self.load_next_sample_seg()
            label_cls = np.array([[[-1]]])
        return im, label_seg, label_cls

    def load_next_sample_seg(self):
        # Did we finish an epoch?
        if self.seg_cur == len(self.indexlist_seg):
            self.seg_cur = 0
            shuffle(self.indexlist_seg)
        index = self.indexlist_seg[self.seg_cur]  # Get the image index
        in_im=self.load_image_seg(index)
        in_label = self.load_label_seg(index)
        self.seg_cur+=1
        return in_im,in_label

    def load_next_sample_cls(self):
        # Did we finish an epoch?
        if self.cls_cur == len(self.indexlist_cls):
            self.cls_cur = 0
            shuffle(self.indexlist_cls)
        index = self.indexlist_cls[self.cls_cur]  # Get the image index
        in_im=self.load_image_cls(index)
        in_label = self.load_label_cls(index)
        self.cls_cur+=1
        return in_im,in_label


    def load_image_seg(self,idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = cv2.imread('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        im = cv2.resize( im, (self.im_shape, self.im_shape), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32, copy=False)
        im -= self.mean
        im = im.transpose((2,0,1))
        return im


    def load_image_cls(self,idx):

        im = cv2.imread(self.cls_dir+'/'+idx.split(' ')[0])
        im = cv2.resize( im, (self.im_shape, self.im_shape), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32, copy=False)
        im -= self.mean
        im = im.transpose((2,0,1))
        return im

    def load_label_seg(self, idx):
        """
        Load label image as 1 x height x width integer array of label indexlist_seg.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
        """
        label = np.load('{}/trainval/{}.npy'.format(self.context_dir, idx))
        label = scipy.misc.imresize(label,(self.im_shape,self.im_shape),'nearest')
        label = label[np.newaxis, ...]
        return label

    def load_label_cls(self, idx):
        label = np.array([int(idx.split(' ')[1])])
        label=label[...,np.newaxis,np.newaxis]
        return label

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'voc_dir', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)

def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
