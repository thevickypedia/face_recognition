import os

import cv2
import face_recognition

# data stored as images in directories train and valid are referred to training and validation dataset
training_dataset, validation_dataset = "train", "valid"

# tolerance level of how easy going or strict the reference should be
learning_rate = 0.6
model = "hog"  # model using which the images are matched

train_faces, train_names = [], []

for character_dir in os.listdir(training_dataset):  # loads the training dataset
    try:
        for file_name in os.listdir(f'{training_dataset}/{character_dir}'):
            # loads all the files within the named repo
            img = face_recognition.load_image_file(f'{training_dataset}/{character_dir}/{file_name}')
            encoded = face_recognition.face_encodings(img)[0]  # generates face encoding matrix
            train_faces.append(encoded)
            train_names.append(character_dir)
    except (IndexError, NotADirectoryError):
        pass

for file_name in os.listdir(validation_dataset):  # loads the validation dataset
    img = face_recognition.load_image_file(f'{validation_dataset}/{file_name}')
    identifier = face_recognition.face_locations(img, model=model)
    encoded_ = face_recognition.face_encodings(img, identifier)  # generates face encoding matrix
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # recognition can only be done on gray scaled images

    for face_encoding, face_location in zip(encoded_, identifier):
        # matches training and validation image matrix and results a boolean expression
        results = face_recognition.compare_faces(train_faces, face_encoding, learning_rate)
        match = None
        if True in results:
            match = train_names[results.index(True)]
            print(match, file_name)
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

    # displays the image and waits until the RETURN key is pressed to display the next matching image
    cv2.imshow(file_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(file_name)
