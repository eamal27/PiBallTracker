import numpy as np
import argparse
import imutils
import cv2

# define HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

counter = 0

camera = cv2.VideoCapture(0)

while True:
    # get current frame
    (grabbed, frame) = camera.read()

    # resize frame
    frame = imutils.resize(frame, width=600)
    # flip frame
    frame = cv2.flip(frame,1)
    # smooth frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create mask for green color
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    # remove blobs in mask
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=5)

    # find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # draw circle around object if radius greater than 10
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.putText(frame, "Tennis Ball Tracker",
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
