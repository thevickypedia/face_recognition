import os

import cv2
import face_recognition

training_dataset, validation_dataset = "train", "valid"

learning_rate = 0.6
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
        results = face_recognition.compare_faces(train_faces, face_encoding, learning_rate)
        match = None
        if True in results:
            match = train_names[results.index(True)]
            print(match, file_name)
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 255]
            cv2.rectangle(img, top_left, bottom_right, color, 3)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(img, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 2)

    cv2.imshow(file_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(file_name)
