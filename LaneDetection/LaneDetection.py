import cv2
import numpy as np
import csv


def Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 100)
    return canny

def ROI(img, scale):
    f = open('../rois.csv', 'r', newline='')
    roi = list(csv.reader(f))
    rois = [[int(float(j) * scale) for j in i] for i in roi]
    triangle = np.array(rois)
    # height = img.shape[0]
    # triangle = np.array([()])
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, triangle, (255, 255, 255))
    masked = cv2.bitwise_and(img, mask)
    return mask, masked

def DisplayLine(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_image

def hstack(imgArray):
    imgConvert = imgArray
    for i in range(len(imgArray)):
        try:
            imgConvert[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)
        except:
            imgConvert[i] = imgArray[i]
    hstack = np.hstack(imgConvert)
    return hstack

