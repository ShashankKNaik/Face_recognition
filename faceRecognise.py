import cv2
import pickle

video = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loaading the face recogniser and the trained data into the program
recognise = cv2.face.LBPHFaceRecognizer_create()
recognise.read("trainner.yml")

labels = {} 

with open("labels.pickle", 'rb') as f:             # reading the stored dictionary during training
    og_label = pickle.load(f)
    labels = {v:k for k,v in og_label.items()}
    print(labels)


while True:
    check,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 5)

    for x,y,w,h in face:
        face_save = gray[y:y+h, x:x+w]
        
        ID, conf = recognise.predict(face_save)    # Predicting the face identified
        conf=int((1-(conf)/300)*100)
        if conf >= 83 :
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),4)
            cv2.putText(frame,labels[ID],(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA )
        else :
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),4)
            cv2.putText(frame,"NotFound",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA )
    cv2.imshow("Video",frame)
    key = cv2.waitKey(100)
    if(key == ord('q')):                           # press 'q' to quit
        break

video.release()
cv2.destroyAllWindows()
