import cv2
import numpy as np
img = cv2.imread('samples/data/shape.jpg')
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(imggray,240,255,cv2.THRESH_BINARY)
contours,_ =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) 
    cv2.drawContours(img,[approx],0,(0,0,0),2)
    x=approx.ravel()[0]
    y=approx.ravel()[1]-5
    if len(approx)==3:
        cv2.putText(img,"Triangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
    if len(approx)==4:
        x1,y1,w,h = cv2.boundingRect(approx)
        aspectratio=float(w)/h
        print(aspectratio)
        if aspectratio >= 0.95 and aspectratio<=1.05:
            cv2.putText(img,"Square",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
        else:
            cv2.putText(img,"Rectangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
    if len(approx)==5:
        cv2.putText(img,"Pentagon",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
    if len(approx)==6:
        cv2.putText(img,"Hegagon",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
    if len(approx)==10:
        cv2.putText(img,"Star",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
    else:
        cv2.putText(img,"Circle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0))
        


# -1 all contours 


cv2.imshow('Shapes',img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()