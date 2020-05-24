import cv2
import numpy as np
img = cv2.imread('samples/data/lena.jpg')


#lr1= cv2.pyrDown(img)
#lr2= cv2.pyrDown(lr1)
#
#hr =cv2.pyrUp(lr2)    #redeuce image inforamtion
#
#cv2.imshow('original img',img)
#cv2.imshow('pyrdown 1 image',lr1)
#cv2.imshow('pyrdown 2 image',lr2)
#cv2.imshow('pyrUP 2 image',hr)


#gausiinan pyramid
layer=img.copy()
gp=[layer]
for i in range(6):
    layer=cv2.pyrDown(layer)
    gp.append(layer)
    #cv2.imshow(str(i),layer)

# laplacian pyramid
layer =gp[5]
cv2.imshow('upper level Gausiian pyramid',layer)
lp =[layer]
for i in range(5,0,-1):
    gausian_extended=cv2.pyrUp(gp[i])
    laplacian=cv2.subtract(gp[i-1],gausian_extended)
    cv2.imshow(str(i),laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()