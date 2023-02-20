import cv2

#Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel()
model.setInputParams(size =(320, 320), scale = 1/255)

#Initialize camara
cap = cv2.VideoCapture(0)

while True:
    #Get frames
    ret, frame = cap.read()

    #Object detection


    cv2.imshow("Frame", frame)
    cv2.waitKey(1)