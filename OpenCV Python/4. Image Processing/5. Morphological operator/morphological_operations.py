import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img =cv.imread('smarties.png',cv.IMREAD_GRAYSCALE)
_, mask = cv.threshold(img,220,255,cv.THRESH_BINARY_INV)

kernal = np.ones((2,2),np.uint8)

dilation = cv.dilate(mask,kernal,iterations=2)
erosion = cv.erode(mask, kernal, iterations=1)
opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernal)
closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernal)
mg = cv.morphologyEx(mask,cv.MORPH_GRADIENT,kernal)
tophat = cv.morphologyEx(mask,cv.MORPH_TOPHAT,kernal)
gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernal)
 
titles =['image','mask','dilation','erosion','opening','closing','mg','tophat','Gradient','blcakhat']
images =[img,mask,dilation,erosion,opening,closing,mg,tophat,gradient,blackhat] 

for i in range(10):
    plt.subplot(2,5,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
plt.show()