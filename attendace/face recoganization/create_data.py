import cv2
import numpy as np
import os
import pyttsx3
import cv2
import os

haar_file = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
datasets = 'datasets'
person_name = input("Enter your name: ")  

path = os.path.join(datasets, person_name)
if not os.path.exists(path):
    os.makedirs(path)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count <= 100:
    print(f"Capturing image {count}...")
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)

    count += 1
    cv2.imshow('Face Capture', img)
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()


face_cascade = load_haar_cascade()

model_file = "face_model.xml"
if not os.path.exists(model_file):
    print("Error: Trained model not found! Run 'train_model.py' first.")
    exit(1)

model = cv2.face.FisherFaceRecognizer_create()
model.read(model_file)

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

names = {}
for (subdirs, dirs, _) in os.walk("datasets"):
    for id, subdir in enumerate(dirs):
        names[id] = subdir

cap = cv2.VideoCapture(0)
print("Face recognition started. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (130, 100))
        label, confidence = model.predict(face_resized)

        if confidence < 800:
            name = names.get(label, "Unknown")
            cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected: {name}")
            engine.say(f"Hello, {name}")
        else:
            name = "Unknown"
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            print("Unknown Person")
            engine.say("Unknown Person Detected")

        engine.runAndWait()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(10) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
