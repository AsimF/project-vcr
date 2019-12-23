# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import requests
# personal imports
from hand_detector import *

#   PARSE SERVER INFO:
parser = argparse.ArgumentParser(description='HAND PROCESS OFFLOADING SERVER')
parser.add_argument('--server_id', type=int, default=-1, help='SERVER_ID: 0 = first/left, 1 = second/right (offload)')
args = parser.parse_args()

#   SET SERVER INFO:
server_id = args.server_id   # this is important to set different server IDs for proper image processing
process_from_height, process_to_height = (0, 320) if (server_id == 0) else (0, 320)  # (320, 480) set this based on the server ID & img resolution manually (or code it to match input)
process_from_width, process_to_width = (0, 240) if (server_id == 0) else (240, 480)  # (320, 480) set this based on the server ID & img resolution manually (or code it to match input)

# load model first thing... (for part 2)
hand_detector = HandDetector('handnet', './handnet.npz', device=0)

# PART 1: This part is for image partitioning algorithm : CONTROL, client splits 50:50 left:right, this server gets left

# const image meta
im_width = 640  # this information must be set in stone! make sure the mobile client uses this as well
im_height = 720  # we will actually use this info instead of calling image_input.size() - checks if wrong size, throw error

# piped image information
im_start_x = 0
im_start_y = 0
im_end_x = 0
im_end_y = 0

# piped_img = np.zeros((im_height, im_width))	# the piped image initialization (for control this will be created after detection)

# data for client (simply x,y for now)
hand_x = 0.0
hand_y = 0.0    #   hand data
hand_z = 0.0
split_start_x = 0
split_start_y = 0   #   next split img
split_end_x = 0
split_end_y = 0

image_input = blank_image = np.zeros((320, 240, 3), np.uint8) 	#  for now just use empty...  # to-do get frame by frame
piped_img = image_input	 # the piped image initialization (for control this will be created after detection)

#   if not image_input.size[:2] == (im_height, im_width):


#   PART 2: PROCESS THE PIPED IMAGE FOR HAND SKELETON -------------- this part is for actual off-load computing
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)


url = "http://192.168.43.1:8080/shot.jpg"

while True:
# this uses the webcam (for testing purposes)
#     ret, img = cap.read()
#     # img = piped_img
#     if not ret:
#         print("Failed to capture image")
#         break

# this uses the phone IP camera (for demo purposes)
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

# extract the correct part of the image from partition algorithm
    piped_img = img[process_from_height:process_to_height, process_from_width:process_to_width]

    hand_keypoints = hand_detector(piped_img)  # be able to send keypoints
    res_img = draw_hand_keypoints(piped_img, hand_keypoints, (0, 0))

    resized_img = cv2.resize(res_img, (1500, 1500), interpolation = cv2.INTER_AREA)  # resize the image to display if not displayed on projector

    cv2.imshow("Control Partition Algorithm - SERVER: " + str(server_id), resized_img)
    cv2.waitKey(1)

#   to-do: recalculate actual position of detections on the original split


#   to-do: send data-for-client to client phone

"""
Code saved for later

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
ap.add_argument("-mW", "--montageW", required=True, type=int,
                help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int,
                help="montage frame height")
args = vars(ap.parse_args())

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["hand"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
CONSIDER = set(["dog", "person", "car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# assign montage width and height so we can view all incoming frames
# in a single "dashboard"
mW = args["montageW"]
mH = args["montageH"]
print("[INFO] detecting: {}...".format(", ".join(obj for obj in
                                                 CONSIDER)))

# start looping over all the frames
while True:
    # receive RPi name and frame from the RPi and acknowledge
    # the receipt
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    # if a device is not in the last active dictionary then it means
    # that its a newly connected device
    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))

    # record the last active time for the device from which we just
    # received a frame
    lastActive[rpiName] = datetime.now()

    # resize the frame to have a maximum width of 400 pixels, then
    # grab the frame dimensions and construct a blob
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # reset the object count for each object in the CONSIDER set
    objCount = {obj: 0 for obj in CONSIDER}

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # detections
            idx = int(detections[0, 0, i, 1])

            # check to see if the predicted class is in the set of
            # classes that need to be considered
            if CLASSES[idx] in CONSIDER:
                # increment the count of the particular object
                # detected in the frame
                objCount[CLASSES[idx]] += 1

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box around the detected object on
                # the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)

    # draw the sending device name on the frame
    cv2.putText(frame, rpiName, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw the object count on the frame
    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
                      objCount.items())
    cv2.putText(frame, label, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # update the new frame in the frame dictionary
    frameDict[rpiName] = frame

    # build a montage using images in the frame dictionary
    montages = build_montages(frameDict.values(), (w, h), (mW, mH))

    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
        cv2.imshow("Home pet location monitor ({})".format(i),
                   montage)

    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF

    # if current time *minus* last time when the active device check
    # was made is greater than the threshold set then do a check
    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        # loop over all previously active devices
        for (rpiName, ts) in list(lastActive.items()):
            # remove the RPi from the last active and frame
            # dictionaries if the device hasn't been active recently
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                lastActive.pop(rpiName)
                frameDict.pop(rpiName)

        # set the last active check time as current time
        lastActiveCheck = datetime.now()

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()

"""