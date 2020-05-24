import cv2
import numpy as np
img = cv2.imread('opencv-logo.png')
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imggray,127,255,0)
contours,herarchy =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print("Number of contours ="+str(len(contours)))
print(contours[0])

# To draw all the contours in an image:
cv2.drawContours(img,contours,-1,(0,255,0),3)
# -1 all contours 

#To draw an individual contour, say 4th contour:
#img = cv2.drawContours(img, contours, 3, (0,255,0), 3)

cv2.imshow('Image',img)
cv2.imsho('gray',imggray)
cv2.waitKey(0)
cv2.destroyAllWindows()