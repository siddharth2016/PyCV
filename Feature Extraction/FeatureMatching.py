"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import numpy as np

if __name__ == "__main__":

    inputImage = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
    targetImage = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    
    descriptor = cv2.ORB_create()
    kp1, des1 = descriptor.detectAndCompute(inputImage, None)
    kp2, des2 = descriptor.detectAndCompute(targetImage, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matcher2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    
    matches = matcher.match(des1, des2)
    mathces = sorted(matches, key=lambda x:x.distance)

    knnmatches = matcher2.knnMatch(des1, des2, k=3)

    normalMatch = cv2.drawMatches(inputImage, kp1, targetImage, kp2, matches[:50], targetImage, flags=2)
    knnMatch = cv2.drawMatchesKnn(inputImage, kp1, targetImage, kp2, knnmatches, targetImage, flags=2)

    cv2.resize(normalMatch, (600, 600))
    
    cv2.imshow("Normal Matches", normalMatch)
    cv2.imshow("KNN Matches", knnMatch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
