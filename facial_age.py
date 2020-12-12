from mtcnn import MTCNN
import tensorflow as tf
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
detector = MTCNN()

model = tf.keras.models.load_model("Age Regression")

while True:
    ret, frame = cap.read()
    
    face = detector.detect_faces(frame)
    if face:
        ages = []
        for i in range(len(face)):
            # if face[i]["box"].shape[0]:
                left,top,width,height = face[i]["box"]
                if width > 0 and height > 0:
                    start_point = (face[i]["box"][0]-30,face[i]["box"][1]-30)
                    end_point = (face[i]["box"][0]+face[i]["box"][2]+30,face[i]["box"][1]+face[i]["box"][3]+30)
                    roi = frame[top-30:top+height+30,left-30:left+width+30]
                    if roi.shape[0] > 0:
                        temp = np.zeros((roi.shape[0],roi.shape[0],3))
                        start = (roi.shape[0] - roi.shape[1])//2
                        end = start + roi.shape[1]
                        temp[::,start:end] = roi
                        temp = cv2.resize(temp,(100,100))
                        temp = temp/255.0
                        cv2.rectangle(frame,start_point,end_point,255,2)
                        age = model.predict(np.array([temp]))
                        ages.append(age)
                        print(int(ages[0]))
    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
print(int(sum(ages)/len(ages)))