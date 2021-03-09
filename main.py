import cv2
import sys

# Retrieve user input
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create haar code
faceCascade = cv2.CascadeClassifier(cascPath)

# Read image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)