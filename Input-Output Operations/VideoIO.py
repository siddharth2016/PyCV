"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np

# Object Oriented Implementation for Video I/O

class Video:

    def __init__(self, videoName = 0):
        self._video = cv2.VideoCapture(videoName)

    def releaseCap(self):
        self._video.release()

    # Display the Video on screen
    def showVideo(self):
        success = self._video.grab()
        _, frame = self._video.retrieve()
        
        while True:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
            success = self._video.grab()
            _, frame = self._video.retrieve()

    # YUV encoded video file to be saved
    def saveYUV(self, fps = 20, size = (420, 240)):
        V = cv2.VideoWriter("YUVencoded.avi", cv2.VideoWriter_fourcc('I','4','2','0'),
                            fps, size)
        _, frame = self._video.read()
        while True:
            cv2.imshow("Video", frame)
            V.write(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                V.release()
                cv2.destroyAllWindows()
                break
            _, frame = self._video.read()

    # MPEG-4 encoded video file to be saved
    def saveMPEG4(self, fps = 20, size = (420, 240)):
        V = cv2.VideoWriter("MPEG4encoded.avi", cv2.VideoWriter_fourcc('X','V','I','D'),
                            fps, size)
        _, frame = self._video.read()
        while True:
            cv2.imshow("Video", frame)
            V.write(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                V.release()
                cv2.destroyAllWindows()
                break
            _, frame = self._video.read()

if __name__ == "__main__":

    V = Video()
    V.showVideo()
    V.saveYUV()
    V.saveMPEG4()
    V.releaseCap()
