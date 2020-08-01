import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image

img1 = io.imread('car.tif')
img2 = io.imread('truck.tif')
img3 = io.imread('van.tif')
img=[img1,img2,img3]
hog_res=[]
for i in range(3):
     
    image = color.rgb2gray(img[i])
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    hog_res.append(hog_image_rescaled)
 

titles =['Image-1','Image-2','Image-3','HOG of Image-1','HOG of Image-2','HOG of Image-3']
images =[img1,img2,img3,hog_res[0],hog_res[1],hog_res[2]] 

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
plt.suptitle('Histogram of Oriented Gradients')
plt.show()