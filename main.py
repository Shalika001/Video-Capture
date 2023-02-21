import cv2

#Opencv DNN
net = cv2.dnn.readNet("./dnn_model/yolov4-tiny.weights", "./dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

#Initialize camara
cap = cv2.VideoCapture(0)

while True:
    #Get frames
    ret, frame = cap.read()

    #Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    print("class_ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)


    cv2.imshow("Frame", frame)
    cv2.waitKey(1)