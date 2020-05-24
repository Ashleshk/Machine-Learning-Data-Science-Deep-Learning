import numpy as np
import cv2 as cv
cap = cv.VideoCapture('vtest.avi')
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
#fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.bgsegm.BackgroundSubtractorGMG()
#fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)
fgbg = cv.createBackgroundSubtractorKNN(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    cv.imshow('Frame', frame)
    cv.imshow('FG MASK Frame', fgmask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv.destroyAllWindows()