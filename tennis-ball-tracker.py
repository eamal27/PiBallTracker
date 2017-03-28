import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import imutils
import cv2

# define HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

counter = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.awb_mode = 'off'
# Start off with ridiculously low gains
rg, bg = (0.5, 0.5)
camera.awb_gains = (rg, bg)
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

# do some white balancing to better detect green color 
with PiRGBArray(camera, size=(128, 72)) as output:
    for i in range(30):
            # Capture a tiny resized image in RGB format, and extract the
            # average R, G, and B values
            camera.capture(output, format='rgb', resize=(128, 72), use_video_port=True)
            r, g, b = (np.mean(output.array[..., i]) for i in range(3))
            # print('R:%5.2f, B:%5.2f = (%5.2f, %5.2f, %5.2f)' % (
            #    rg, bg, r, g, b))
            # Adjust R and B relative to G, but only if they're significantly
            # different (delta +/- 2)
            if abs(r - g) > 2:
                if r > g:
                    rg -= 0.1
                else:
                    rg += 0.1
            if abs(b - g) > 1:
                if b > g:
                    bg -= 0.1
                else:
                    bg += 0.1
            camera.awb_gains = (rg, bg)
            output.seek(0)
            output.truncate()



for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # get current frame
    frame = image.array

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
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=1)

    # find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
	for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if (M["m00"] != 0) :
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # draw circle around object if radius greater than 10
            if radius > 10 and radius < 100:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
    rawCapture.truncate(0)

    if key == ord("q"):
        break
