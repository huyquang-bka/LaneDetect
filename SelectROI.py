import cv2
import csv

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)
VIDEO_SOURCE_PATH = "Pexels Videos 4516.mp4"

cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
suc, image = cap.read()

cv2.imwrite("frame0.jpg", image)
cap.release()

cv2.namedWindow("frame0", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame0", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

point = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      point.append([x,y])
      with open('rois.csv', 'w', newline='') as outf:
        csvw = csv.writer(outf)
        csvw.writerows(point)
      outf.close()
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, str(x)+','+str(y), (x+10,y), font, 0.5, (0,0,255), 2)
      cv2.circle(img,(x,y),2,(0,0,255),5)
      cv2.imshow("image", img)

img = cv2.imread("frame0.jpg")

# r = cv2.selectROIs('ROI Selector', img, showCrosshair=False, fromCenter=False)

cv2.imshow("image", img)

#calling the mouse click event
cv2.setMouseCallback("image", click_event)

# rlist = point.tolist()
# print(rlist)



cv2.waitKey()
cv2.destroyAllWindows()

