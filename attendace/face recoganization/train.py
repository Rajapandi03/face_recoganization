import cv2
import numpy as np
import os

datasets = 'datasets'
model_file = 'face_model.xml'
haar_file = 'haarcascade_frontalface_default.xml'

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for subdir in os.listdir(datasets):
    names[id] = subdir
    subject_path = os.path.join(datasets, subdir)

    for filename in os.listdir(subject_path):
        path = os.path.join(subject_path, filename)
        images.append(cv2.imread(path, 0))
        labels.append(id)
    
    id += 1

(images, labels) = [np.array(i) for i in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.save(model_file)

print("Training complete. Model saved as", model_file)
