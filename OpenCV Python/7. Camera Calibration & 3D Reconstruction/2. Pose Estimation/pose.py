import cv2
import numpy as np
import glob

# Load previously saved data
with np.load('1. Camera Calibration\camera.py') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    