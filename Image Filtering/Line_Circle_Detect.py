"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np
from scipy import ndimage
import imutils

class Detect:

    def __init__(self, imageName):
        self.image = cv2.imread(imageName, cv2.IMREAD_COLOR)
        self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # Detect Hough Circles
    def detectCircle(self, minRad = 0, maxRad = 0):
        img = self.image.copy()
        circles = cv2.HoughCircles(self.grayimage, cv2.HOUGH_GRADIENT,
                                   1, 120, param1 = 100, param2 = 30, minRadius = minRad,
                                   maxRadius = maxRad)
        circles =  np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0,255,0), 2)

        return img

    # Detect Hough Lines
    def detectLine(self, minLen = 20, maxLGap = 5):
        img = self.image.copy()
        edgeimg = cv2.Canny(img, 100, 200)
        lines = cv2.HoughLinesP(edgeimg, 1, np.pi/180, minLen,
                                maxLGap)
        for line in lines[0]:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,0,255), 2)

        return img

if __name__ == "__main__":
    I = Detect("1.jpg")
    circle = I.detectCircle()
    line = I.detectLine()

    circle = imutils.resize(circle, 400)
    line = imutils.resize(line, 400)
    
    cv2.imshow("Circles", circle)
    cv2.imshow("Lines", line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
