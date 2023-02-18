import cv2

cpa = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)