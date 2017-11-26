import cv2
import constants as C
from skimage.feature import hog


def getHogVector(roi):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    H, hoggedImage = hog(roi, visualise=True)
    return H,hoggedImage


def drawAndGetRoi(frame):
    (x0, y0) = (C.x_roi, C.y_roi)
    (widthRoi, heightRoi) = (C.widthRoi, C.heightRoi)
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + widthRoi, y0 + heightRoi), (0, 0, 255), 2)
    roi = frame[y0:y0+heightRoi, x0:x0+widthRoi]
    return roi, frame
