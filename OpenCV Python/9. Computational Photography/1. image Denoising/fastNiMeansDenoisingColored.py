import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Noise_salt_and_pepper.png')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
