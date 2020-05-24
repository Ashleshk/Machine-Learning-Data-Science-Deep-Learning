import numpy as np
import cv2

img = cv2.imread('star.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)


#Image moments help you to calculate some features 
#like center of mass of the object, area of the 
#object etc. 
cnt = contours[0]
M = cv2.moments(cnt)
print(M)

#From this moments, you can extract useful data 
#like area, centroid etc. 
#Centroid is given by the relations, C_x = \frac{M_{10}}{M_{00}} and C_y = \frac{M_{01}}{M_{00}}.
# This can be done as follows:

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#Contour Area
#Contour area is given by the function cv2.contourArea() or from moments, M[‘m00’].

#cOUNTOR PERIMETER
perimeter = cv2.arcLength(cnt,True)

# Contour Approximation
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

# CONVEX HULL
#cv2.convexHull() function checks a curve for convexity defects and corrects it. Generally
# speaking, convex curves are the curves which are always bulged out, or at-least flat.
#hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    #Arguments details:

        #points are the contours we pass into.
        #hull is the output, normally we avoid it.
        #clockwise : Orientation flag. If it is True, the output convex hull is oriented clockwise. Otherwise, it is oriented counter-clockwise.
        #returnPoints : By default, True. Then it returns the coordinates of the hull points. If False, it returns the indices of contour points corresponding to the hull points.
hull = cv2.convexHull(cnt)





# Checking Convexity
k = cv2.isContourConvex(cnt)


#         Bounding Rectangle
#There are two types of bounding rectangles.

#7.a. Straight Bounding Rectangle
#It is a straight rectangle, it doesn’t consider the rotation of the object. So area of the bounding rectangle won’t be minimum. It is found by the function cv2.boundingRect().

#Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#7.b. Rotated Rectangle
#Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following detals - ( top-left corner(x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. 
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,(0,0,255),2)


#   Minimum Enclosing Circle
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)

















