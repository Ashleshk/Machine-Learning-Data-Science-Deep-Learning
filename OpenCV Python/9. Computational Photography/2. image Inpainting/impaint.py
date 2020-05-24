import numpy as np
import cv2

img = cv2.imread('messi5.jpg')
mask = cv2.imread('mask.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
