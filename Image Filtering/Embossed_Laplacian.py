"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np
from scipy import ndimage
import imutils

class Filters:

    def __init__(self, imageName):
        self.image = cv2.imread(imageName, cv2.IMREAD_COLOR)
        self.image = imutils.resize(self.image, 300, 600)
        self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.kernelemboss = np.array([[-2, -1, 0],
                                      [-1, 1, 1],
                                      [0, 1, 2]])

    def Embossed(self):
        emboss = cv2.filter2D(self.image, -1, self.kernelemboss)
        return emboss

    def Laplace(self):
        laplace = cv2.Laplacian(self.grayimage, cv2.CV_8U, self.grayimage,
                                ksize = 7)
        inverseAlpha = (1.0/255)*(255 - laplace)
        channels = cv2.split(self.image)
        for channel in channels:
            channel[:] = channel*inverseAlpha
        laplace = cv2.merge(channels)
        return laplace

    def MedianBlur(self):
        blur = cv2.medianBlur(self.image, ksize = 5)
        return blur

if __name__ == "__main__":
    I = Filters("3.jpg")
    emb = I.Embossed()
    med = I.MedianBlur()
    lap = I.Laplace()

    cv2.imshow("Embossed", emb)
    cv2.imshow("Laplacian", lap)
    cv2.imshow("Median Blur", med)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
