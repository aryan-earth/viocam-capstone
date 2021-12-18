# Imports
import numpy as np
from cv2 import cv2 as cv
from numpy.core.fromnumeric import resize, sort
from termcolor import colored
import argparse
from time import sleep

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# License Plate
license_plate_weights_path = "/home/glitch101/capstone_project/license_plate_weights/lp.weights"
license_plate_config_path = "/home/glitch101/capstone_project/license_plate_weights/lp.cfg"
license_plate_net = cv.dnn.readNetFromDarknet(license_plate_config_path, license_plate_weights_path)

def find_license_plate(frame):
    (H, W) = frame.shape[:2]

    layer_names = license_plate_net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in license_plate_net.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    license_plate_net.setInput(blob)
    layer_outputs = license_plate_net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            box = detection[0:4] * np.array([W, H, W, H])
            (center_x, center_y, width, height) = box.astype('int')

            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            boxes.append([x, y, int(width), int(height), int(center_x), int(center_y)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3) # Last two args are confidence and threshold respectively. 
    
    # Check if idxs is empty
    if (len(idxs) == 0):
        raise  Exception(colored('No license plates detected', 'red'))
        return -1
    else:
        # License plate found
        license_plate_detection_image = frame.copy()
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (center_x, center_y) = (boxes[i][4], boxes[i][5])

            cv.rectangle(license_plate_detection_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #note DON'T forget this coordinate system you idiot
            license_plate = image[y:y+h, x:x+w]
            cv.imwrite("/home/glitch101/capstone_project/results/license_plate{}.jpg".format(i), license_plate)
        return license_plate_detection_image


image = cv.imread(args["image"])
license_plate_detection_image = find_license_plate(image)

# Save image
cv.imwrite("/home/glitch101/capstone_project/results/result.jpg", license_plate_detection_image)
print("Done")