import numpy as np
import cv2
import _init_paths
import caffe

class CaffeForward:

    def __init__(self, net_def, weights, mean_file,batch_size):
        #self.im_shape=224
        self.net_ = caffe.Net(net_def, weights, caffe.TEST)
        n_channels, height, width = [x for x in self.net_.blobs['data'].shape][-3:]
        self.im_shape=height
        self.net_.blobs['data'].reshape(batch_size, n_channels, height, width)

        if mean_file.endswith('npy'):
            self.mean = np.load(mean_file)[0].mean(1).mean(1)
        else:
            with open(mean_file, 'r') as f:
                self.mean = np.array([float(x) for x in f])

    def forward(self, img_paths):
        for i, img_path in enumerate(img_paths):
            im = cv2.imread(img_path)
            im = cv2.resize( im, (self.im_shape, self.im_shape), interpolation=cv2.INTER_LINEAR)
            im = im.astype(np.float32, copy=False)
            im -= self.mean
            im = im.transpose((2,0,1))

            self.net_.blobs['data'].data[i,...] = im
        self.net_.forward()

    def get_blob(self, blob_name):
        return self.net_.blobs[blob_name].data

    def get_params(self, layer_name):
        return self.net_.params[layer_name]

if __name__ == '__main__':

    #if len(sys.argv) < 5:
    #    print('Use: python cnn_fe.py [model_file] [model_weights] [model_mean] [image_list]')
    #    sys.exit(0)

    #model_file = sys.argv[1]
    #model_weights = sys.argv[2]
    #model_mean = sys.argv[3]
    #img_path = sys.argv[4]
    #gpu_id = int(sys.argv[5])

    model_file = 'deploy.prototxt'
    model_weights = 'model.caffemodel'
    model_mean = 'mean.txt'
    img_list = 'test.txt'
    dataroot='data'

    gpu_id = 0
    dstf=open('out.txt','a')


    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    batch_size=10
    cf = CaffeForward(model_file, model_weights, model_mean, batch_size)

    with open(img_list, 'r') as f:
        lines = f.readlines()
    img_paths=[]
    labels=[]
    cnt = 0
    for line in lines:
        line=line.rstrip()
        img_path = dataroot+'/'+line.split()[0]
        img_paths.append(img_path)
        label = line.split()[1]
        labels.append(label)
        if len(img_paths)==batch_size:
            cnt += batch_size
            print 'processing', cnt
            cf.forward(img_paths)
            probs = cf.get_blob('prob')
            for i in range(batch_size):
                outline = ' '.join([img_paths[i]]+[labels[i]]+[str(x) for x in probs[i]])
                #print outline
                dstf.write(outline+'\n')
            img_paths=[]
            labels=[]

        if cnt + len(img_paths) == len(lines):
            cf.forward(img_paths)
            probs = cf.get_blob('prob')
            for i in range(len(img_paths)):
                outline = ' '.join([img_paths[i]]+[labels[i]]+[str(x) for x in probs[i]])
                dstf.write(outline+'\n')

