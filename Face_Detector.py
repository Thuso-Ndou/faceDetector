import cv2
from random import randrange

# loaded some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
#img = cv2.imread('RD.jpg')

#capture video from webcame
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # read the current frame
    successful_frame_read, frame = webcam.read()

    #convert the image to grayscale
    grayscaled_img =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, randrange(256), 0), 2)

    cv2.imshow('Huso Face Detector', frame)
    key = cv2.waitKey(1)

    #stop if Q key is pressed
    if key==81 or key==113:
        break

# release the video capture object
webcam.release()

print("Code Complete")