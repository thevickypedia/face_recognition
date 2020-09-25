import os

import cv2
import face_recognition

training_dataset, validation_dataset = "train", "valid"

training_rate = 0.6
model = "hog"

train_faces, train_names = [], []

for character_dir in os.listdir(training_dataset):
    for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
        img = face_recognition.load_image_file(f'{training_dataset}/{character_dir}/{file_name}')
        encoded = face_recognition.face_encodings(img)[0]
        train_faces.append(encoded)
        train_names.append(character_dir)

for file_name in os.listdir(validation_dataset):
    img = face_recognition.load_image_file(f'{validation_dataset}/{file_name}')
    identifier = face_recognition.face_locations(img, model=model)
    encoded_ = face_recognition.face_encodings(img, identifier)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoded_, identifier):
        results = face_recognition.compare_faces(train_faces, face_encoding, training_rate)
        match = None
        if True in results:
            match = train_names[results.index(True)]
            print(match, file_name)
