import face_recognition
import os

training_dataset = "train"

train_faces = []

for character_dir in os.listdir(training_dataset):
    for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
        img = face_recognition.load_image_file(f'{training_dataset}/{character_dir}/{file_name}')
        encoded = face_recognition.face_encodings(img)[0]
        train_faces.append(encoded)

print(train_faces)
