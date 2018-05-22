"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np

class featureExt(object):
    """ Implements various feature extraction methods provided by opencv3 """

    def __init__(self, imageName):

        self.img = cv2.imread(imageName, cv2.IMREAD_COLOR)
        self.img = cv2.resize(self.img, (300, 300))
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detectCorners(self):
        """ Implements cornerHarris method of opencv3 """

        img = self.img.copy()
        corners = cv2.cornerHarris(np.float32(self.gray), 2, 23, 0.04)
        img[corners>0.01*corners.max()] = [0, 0, 255]

        return img

    def detectSIFT(self):
        """ Implements SIFT_create() method of opencv3 """

        img = self.img.copy()
        siftobj = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptor = siftobj.detectAndCompute(self.gray, None)

        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                flags=4, color=(50, 160, 230))

        return img

    def detectSURF(self, hessian = 5000):
        """ Implements SURF_create() method of opencv3 """

        img = self.img.copy()
        surfobj = cv2.xfeatures2d.SURF_create(float(hessian))
        keypoints, descriptor = surfobj.detectAndCompute(self.gray, None)

        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                flags=4, color=(50, 160, 230))

        return img

if __name__ == "__main__":

    imgObj = featureExt("1.jpg")

    imgcorners = imgObj.detectCorners()
    imgSIFT = imgObj.detectSIFT()
    imgSURF = imgObj.detectSURF()

    while True:
        cv2.imshow("corners", imgcorners)
        cv2.imshow("Sift", imgSIFT)
        cv2.imshow("Surf", imgSURF)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
    
