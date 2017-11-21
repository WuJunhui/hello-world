#
import cv2
import matplotlib.pyplot as plt

#Display a cv2.imread() image in plt
im=cv2.imread('1.jpg')
fig0=plt.figure(0,figsize=(6,10))#(w,h) inch
plt.axis('off')
plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
plt.show()
fig0.clear()

#cv2 rectangle
xmin,ymin,xmax,ymax=100,100,200,400
im_copy=im.copy()
cv2.rectangle(im_copy,(xmin,ymin),(xmax,ymax),(255,255,255),thickness=-1)
cv2.putText(im_copy,'cute cat',(xmin,ymax),cv2.FONT_HERSHEY_COMPLEX,2, (0,0,255),2)
cv2.imshow('im',im)
cv2.imshow('rectim',im_copy)
cv2.waitKey(0)
cv2.destroyWindow('rectim')
cv2.destroyAllWindows()

