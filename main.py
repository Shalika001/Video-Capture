import cv2

#Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")

#Initialize camara
cap = cv2.VideoCapture(0)

while True:
    #Get frames
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)