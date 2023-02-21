import cv2

#Opencv DNN
net = cv2.dnn.readNet("./dnn_model/yolov4-tiny.weights", "./dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

#Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object list")
print(classes)

#Initialize camara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 740)

#Create Window
cv2.namedWindow("Frame")
#cv2.setMouseCallback("Frame", click_button)

while True:
    #Get frames
    ret, frame = cap.read()

    #Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_ids, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) =bbox
        class_name = classes[class_ids]

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    print("class_ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)