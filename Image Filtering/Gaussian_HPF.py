"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np
from scipy import ndimage

class Filters:

    def __init__(self, imageName):
        self.image = cv2.imread(imageName, 0)
        self.kernel3 = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
        self.convoluted = None
        self.gaussian = None
        
    def HPF(self):
        self.convoluted = ndimage.convolve(self.image, self.kernel3)
        return self.convoluted

    def Gauss(self):
        self.gaussian = cv2.GaussianBlur(self.image, (13,13), 0)
        return self.gaussian

    def Filtered(self):
        filtered = self.image - self.gaussian
        return filtered

if __name__ == "__main__":

    I = Filters("1.jpg")
    hpf = I.HPF()
    gauss = I.Gauss()
    filtered = I.Filtered()
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
