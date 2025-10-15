import cv2
import os
import time
# Import numpy for matrices calculations
import numpy as np
import time
import pandas as pd

# Create Local Binary Patterns Histograms for face recognization
##recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')
##recognizer.read('/home/pi/Desktop/face_recog_folder/Raspberry-Face-Recognition-master/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)
df=pd.read_csv('names.csv')

while True:
        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)


        # For each face in faces
        for(x,y,w,h) in faces:

            # Create rectangle around the face
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)

            # Recognize the face belongs to which ID
            Id,i = recognizer.predict(gray[y:y+h,x:x+w])

            print(Id, i)

            if i < 60:
                name=df.loc[(df['id']==Id)]['name'].values[0]
                cv2.putText(im, name, (x,y-40), font, 2, (255,255,255), 3)
            else:
                cv2.putText(im, "unknown", (x,y-40), font, 2, (255,255,255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im)
        
        # If 'q' is pressed, close program
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
           
cam.release()
# Close all windows
cv2.destroyAllWindows()
