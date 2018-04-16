import sys
import os
import numpy
sys.path.append('/opt/caffe/python')
import caffe

mean_bgr = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
width = int(sys.argv[1])
height = int(sys.argv[2])
output_name = sys.argv[6]

mean_arr = numpy.empty((1,3,width,height))
mean_arr[0, 0, ...] = mean_bgr[0]
mean_arr[0, 1, ...] = mean_bgr[1]
mean_arr[0, 2, ...] = mean_bgr[2]

#print(mean_arr[0][0])
#print(mean_arr[0][1])
#print(mean_arr[0][2])

mean_blob = caffe.io.array_to_blobproto(mean_arr)
binaryproto_file = open('{}.binaryproto'.format(output_name), 'wb' )
binaryproto_file.write(mean_blob.SerializeToString())
binaryproto_file.close()
numpy.save('{}.npy'.format(output_name), mean_arr)
