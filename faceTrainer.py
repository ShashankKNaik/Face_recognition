import cv2
import os
import numpy as np
import pickle

recognise = cv2.face.LBPHFaceRecognizer_create()

def getdata():
    current_id = 0
    label_id = {} 
    face_train = [] 
    face_label = [] 

    for root,dirs,files in os.walk('image_data'):        
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)        # path to the each image
                label = os.path.basename(root).lower() # assigning lable as image folder name
                
                if not label in label_id:              # adding ID to each unique lable
                    label_id[label] = current_id
                    current_id += 1

                ID = label_id[label]
                image_mat = cv2.imread(path)	       # reading each image data
                img=cv2.cvtColor(image_mat,cv2.COLOR_BGR2GRAY)
                cv2.imshow("Test",img)
                cv2.waitKey(10)
                face_train.append(img)
                face_label.append(ID)

    with open("labels.pickle", 'wb') as f:             # saving label_id to file
        pickle.dump(label_id, f)
   
    return face_train,face_label

face,ids = getdata()
recognise.train(face, np.array(ids))		       # training and saving it in .yml file
recognise.save("trainner.yml")