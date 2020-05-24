# accessing RED value
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
img.item(10,10,2)


# modifying RED value
img.itemset((10,10,2),100)
img.item(10,10,2)


#       Accessing Image Properties
print(img.shape)

#Total number of pixels is accessed by img.size:
print(img.size)

#Image datatype is obtained by
print(img.dtype)



#         IMAGE ROI
# ROI is again obtained using Numpy indexing.

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
cv2.imshow('img',img)

#     SPLITTING AND MERGING IMAGE CHANNELS

b,g,r = cv2.split(img)

#Or
b = img[:,:,0]

img = cv2.merge((b,g,r))



#   Making Borders for Images (Padding)

import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

img1 = cv2.imread('opencv-logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()





















