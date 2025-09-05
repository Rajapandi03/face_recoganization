import cv2
import numpy as np
import os
import pandas as pd
import pyttsx3
from datetime import datetime


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

haar_file = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
model_file = 'face_model.xml'
excel_file = 'attendance.xlsx'

if not os.path.exists(model_file):
    print("Error: Trained model not found! Run 'train_model.py' first.")
    exit(1)

face_cascade = cv2.CascadeClassifier(haar_file)
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_file)

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Get existing names for today's attendance
today = datetime.today().strftime('%Y-%m-%d')
existing_names = set(df[df['Date'] == today]['Name'])

webcam = cv2.VideoCapture(0)
names = {id: name for id, name in enumerate(os.listdir('datasets'))}

print("Face Recognition Started. Press 'Esc' to exit.")

while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (130, 100))

        label, confidence = model.predict(face_resize)

        if confidence < 80:  
            name = names.get(label, "Unknown")
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if name not in existing_names:  
                time_now = datetime.now().strftime('%H:%M:%S')
                df.loc[len(df)] = [name, today, time_now]
                df.to_excel(excel_file, index=False)

                engine.say(f"{name} detected. Attendance marked.")
                engine.runAndWait()
                
                existing_names.add(name)
        
        else:
            cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) == 27:  
        break

webcam.release()
cv2.destroyAllWindows()

print("Attendance saved in", excel_file)
