"""
Email: siddharthchandragzb@gmail.com
"""

# Library Imports
import cv2
import numpy as np

# Class FaceDetect

class FaceDetect(object):
    """ This class will contains methods like detectEyes, detectFace and detectSmile """

    def __init__(self, imageName):
        self.img = cv2.imread(imageName, cv2.IMREAD_COLOR)
        self.__grayimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detectFace(self, cascade = "./cascades/haarcascade_frontalface_default.xml", window = "Faces"):
        """ This will draw green rectangle over the detected face """
        
        face_cascade = cv2.CascadeClassifier(cascade)
        faces = face_cascade.detectMultiScale(self.__grayimg, 1.3, 5)
        img = self.img.copy()
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.namedWindow(window)
        cv2.imshow(window, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return None
    
    def detectEyes(self, cascade = "./cascades/haarcascade_eye.xml", window = "Eyes", face = False):
        """ This will draw rectangle over eyes detected, will draw over face also if face variable set to True """

        face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(self.__grayimg, 1.3, 5)
        eye_cascade = cv2.CascadeClassifier(cascade)
        img = self.img.copy()
        for (x,y,w,h) in faces:
            if face:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ROI = self.__grayimg[y:y+h, x:x+w].copy()

            eyes = eye_cascade.detectMultiScale(ROI, 1.03, 5, 0, (30, 30))

            for (ex,ey,ew,eh) in eyes:
                img = cv2.rectangle(img, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)

        cv2.namedWindow(window)
        cv2.imshow(window, img)
        #cv2.imshow("roi", ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return None

    def detectSmile(self, cascade = "./cascades/haarcascade_smile.xml", window = "Simle"):
        """ This will draw rectangle over smiling face """

        face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(self.__grayimg, 1.3, 5)
        eye_cascade = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")
        smile_cascade = cv2.CascadeClassifier(cascade)
        img = self.img.copy()

        for (x,y,w,h) in faces:

            ROI = self.__grayimg[y:y+h, x:x+w].copy()
            eyes = eye_cascade.detectMultiScale(ROI, 1.03, 5, 0, (30,30))

            for (ex,ey,ew,eh) in eyes:

                ROI2 = ROI[ey+eh:ey+h, ex:ex+w]

                smiles = smile_cascade.detectMultiScale(ROI2, 1.03, 5, 0, (60,60))

                for (sx,sy,sw,sh) in smiles:
                    # Smile area is on face and below eyes
                    img = cv2.rectangle(img, (x+ex+sx,y+eh+ey+sy), (x+ex+sx+sw,y+eh+ey+sy+sh), (0,255,0), 2)

        cv2.namedWindow(window)
        cv2.imshow(window, img)
        #cv2.imshow("roi", ROI2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return None

    @property
    def grayimg(self):
        return self.__grayimg
        

if __name__ == "__main__":

    detect = FaceDetect("face-smile.jpg")
    detect.detectFace()
    detect.detectEyes()
    detect.detectSmile()
    detect.detectEyes(face = True)
    gray = detect.grayimg
    print(gray)
            
