import face_recognition
import os

training_dataset, validation_dataset = "train", "valid"
model = 'hog'

train_faces, train_names = [], []

for character_dir in os.listdir(training_dataset):
    for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
        img = face_recognition.load_image_file(f'{training_dataset}/{character_dir}/{file_name}')
        encoded = face_recognition.face_encodings(img)[0]
        train_faces.append(encoded)

for file_name in os.listdir(validation_dataset):
    img = face_recognition.load_image_file(f'{validation_dataset}/{file_name}')
    identifier = face_recognition.face_locations(img, model=model)
    encoded_ = face_recognition.face_encodings(img, identifier)
    print(encoded_)
