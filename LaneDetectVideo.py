import cv2
import numpy as np
import csv

scale = 0.5
path_video = 'Pexels Videos 4516.mp4'
cap = cv2.VideoCapture(path_video)

def Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def ROI(img, scale):
    f = open('rois.csv', 'r', newline='')
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
            imgConvert[i] = cv2.cvtColor(imgArray[i],cv2.COLOR_GRAY2BGR)
        except:
            imgConvert[i] = imgArray[i]
    hstack = np.hstack(imgConvert)
    return hstack
while True:
    img = cap.read()[1]
    img = cv2.resize(img, dsize=None, dst=None, fx=scale, fy=scale)
    lane_img = img.copy()
    canny = Canny(img)
    mask, crop_image = ROI(canny, scale)
    lines = cv2.HoughLinesP(crop_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = DisplayLine(img, lines)
    combo_img = cv2.addWeighted(lane_img,1,line_image,1,0)

    hStack1 = hstack([img,combo_img])
    hStack2 = hstack([canny,crop_image])
    vStack = np.vstack([hStack1,hStack2])

    cv2.imshow('All', vStack)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
