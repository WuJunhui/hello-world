#coding: utf-8
import os
import numpy
import cv2

input_path = 'data2det1027'
persp_txt='cam0.persp'


image_list = [os.sep.join([input_path, x]) for x in os.listdir(input_path) if x.lower().endswith('.jpg')]

# load perspective params
with open(persp_txt, 'r') as f:
    line = f.readline()
    corners = numpy.array([float(x) for x in line.split()], dtype=numpy.float32)
    corners = corners.reshape((4, 2))
    line = f.readline()
    rect_corners = numpy.array([float(x) for x in line.split()], dtype=numpy.float32)
    rect_corners = rect_corners.reshape((4, 2))

m_pers = cv2.getPerspectiveTransform(corners, rect_corners)

# correct & display corrected test images
dstdir='a_'+input_path
if not os.path.exists(dstdir):
    os.mkdir(dstdir)
for i, filepath in enumerate(image_list):
    print(i, filepath)
    if not os.path.exists(filepath):
        print 'no file exists!'
    test_img = cv2.imread(filepath)
    cv2.imshow('ori',test_img)
    test_corrected = cv2.warpPerspective(test_img, m_pers, (1280, 720))
    dstpath='a_'+filepath.split('.jpg')[0]+'_a.jpg' 
    cv2.imwrite(dstpath, test_corrected)

    cv2.imshow('test', test_corrected)

    cv2.waitKey(0)
cv2.destroyAllWindows()
print 'Done!'
