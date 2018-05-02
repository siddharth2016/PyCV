"""

Email: siddharthchandragzb@gmail.com

"""
# Object Oriented Image Input/Output

import cv2
import numpy as np

class Image:

    def __init__(self, imgName, encode = 0):
        self._image = cv2.imread(imgName, encode)
        self._Himage = None
        self._Vimage = None

    # Horizontal Flip of the Image    
    def flipH(self):
        self._Himage = cv2.flip(self._image, 0)
        return self._Himage

    #Vertical Flip of the Image
    def flipV(self):
        self._Vimage = cv2.flip(self._image, 1)
        return self._Vimage

    #Show the Images
    def show(self):
        cv2.imshow("Image", self._image)
        if self._Himage.all() != None:
            cv2.imshow("Horizontal flip Image", self._Himage)
        if self._Vimage.all() != None:
            cv2.imshow("Vertical flip Image", self._Vimage)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Save Images on disk
    def save(self, Himg = 0, Vimg = 0):
        cv2.imwrite("GrayImage.png", self._image)
        if Himg:
            cv2.imwrite("HorizontalFlipped.png", self._Himage)
        if Vimg:
            cv2.imwrite("VerticalFlipped.png", self._Vimage)

if __name__ == "__main__":

    I = Image("1.jpg")
    horizontalImage = I.flipH()
    verticalImage = I.flipV()
    I.show()
    I.save(Himg = 1, Vimg = 1)
    
    
