import cv2
import os

video = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # to detect face
i=0
offset=50
name=input('Enter your name : ')
os.mkdir("image_data/"+name)                          # create folder with name to store image

while True:
    check, frame =video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       # convert to grayscale image
    faces=detector.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5) # getting the face position in frame
    for(x,y,w,h) in faces:
        i=i+1
        captured_face = frame[y:y+h,x:x+w]
        face = cv2.resize(captured_face,(200,200))    # resizing the captured face
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("image_data/"+name+"/"+name+"-"+ str(i) + ".jpg",face) # saving image in name folder
        cv2.imshow('frame',frame[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(250)
    if i>20:
        video.release()
        cv2.destroyAllWindows()
        break

print("collecting samples complete")