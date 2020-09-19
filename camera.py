import cv2

# All packages contain haarcascade files. cv2.data.haarcascades can be used as a shortcut to the data folder.
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
print(capture.read())
