"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np
from scipy import ndimage
import imutils

class Segment:

    def __init__(self, imageName):
        self.image = cv2.imread(imageName, cv2.IMREAD_COLOR)
        self.image = imutils.resize(self.image, 600)
        self.grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def grabcutSegment(self):
        img = self.image.copy()
        mask = np.zeros(img.shape[:2], dtype = np.uint8)

        foregroundGMM = np.zeros((1, 65), dtype = np.float64)
        backgroundGMM = np.zeros((1, 65), dtype = np.float64)

        rect = (85, 80, 320, 420)   # Optimum result, for naruto image used

        cv2.grabCut(img, mask, rect, backgroundGMM, foregroundGMM,
                    6, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2) | (mask==0), 0 ,1).astype('uint8')
        img = img*mask2[:, :, np.newaxis]
        #print(mask)
        return img

    def watershedSegment(self):
        img = self.image.copy()
        ret, thresh = cv2.threshold(self.grayimage,0,255,
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,
                                   kernel,iterations = 2)
        
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg =cv2.threshold(dist_transform,
                                    0.7*dist_transform.max(),255,0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers+1
        markers[unknown==255] = 0

        markers = cv2.watershed(img,markers)
        img[markers == -1] = [0,255,0]

        return img

if __name__ == "__main__":
    I = Segment("1.jpg")
    GBC = I.grabcutSegment()
    WTR = I.watershedSegment()
    cv2.imshow("GBC", GBC)
    cv2.imshow("WTR", WTR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
