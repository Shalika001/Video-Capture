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
    for class_ids, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) =bbox
        #print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    print("class_ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)