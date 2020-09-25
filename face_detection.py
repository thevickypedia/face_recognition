import cv2

# All packages contain haarcascade files. cv2.data.haarcascades can be used as a shortcut to the data folder.
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)

while True:
    try:
        ignore, image = capture.read()
        # convert the captured image to grayscale
        scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale_factor = 1.1  # Parameter specifying how much the image size is reduced at each image scale.
        min_neighbors = 5  # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        faces = cascade.detectMultiScale(scale, scale_factor, min_neighbors)
        frame_thickness = 3
        frame_color = (255, 255, 255)
        for side1, side2, side3, side4 in faces:
            cv2.rectangle(image, (side1, side2), (side1 + side3, side2 + side4), frame_color, frame_thickness)
        cv2.imshow('image', image)
        cv2.waitKey(30) & 0xff
    except KeyboardInterrupt:
        capture.release()
        break
