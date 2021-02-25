import cv2
import sys

# Retrieve user input
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create haar code
faceCascade = cv2.CascadeClassifier(cascPath)