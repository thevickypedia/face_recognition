import cv2

# All packages contain haarcascade files. cv2.data.haarcascades can be used as a shortcut to the data folder.
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
ignore, image = capture.read()

cv2.imshow('image', image)
cv2.waitKey(30) & 0xff
