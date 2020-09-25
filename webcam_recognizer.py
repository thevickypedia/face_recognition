import os

import cv2
import face_recognition

# data stored as images in directories train and valid are referred to training and validation dataset
training_dataset = "train"
validation_video = cv2.VideoCapture(0)

# tolerance level of how easy going or strict the reference should be
learning_rate = 0.6
model = "hog"  # model using which the images are matched

train_faces, train_names = [], []

for character_dir in os.listdir(training_dataset):  # loads the training dataset
    for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
        # loads all the files within the named repo
        img = face_recognition.load_image_file(f'{training_dataset}/{character_dir}/{file_name}')
        try:
            encoded = face_recognition.face_encodings(img)[0]  # generates face encoding matrix
            train_faces.append(encoded)
            train_names.append(character_dir)
        except IndexError:
            pass

while True:
    ret, img = validation_video.read()
    identifier = face_recognition.face_locations(img, model=model)
    encoded_ = face_recognition.face_encodings(img, identifier)  # generates face encoding matrix
    for face_encoding, face_location in zip(encoded_, identifier):
        # matches training and validation image matrix and results a boolean expression
        results = face_recognition.compare_faces(train_faces, face_encoding, learning_rate)
        match = None
        if True in results:
            match = train_names[results.index(True)]
            # Draws a rectangle around the face
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 255]
            cv2.rectangle(img, top_left, bottom_right, color, 3)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(img, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 2)

    # displays the webcam image and waits until the 'e' key is pressed to exit
    cv2.imshow(None, img)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
