# Find helmets in an image and return cropped out image of every helmet detected in that image.

# Imports
import numpy as np
from cv2 import cv2 as cv
from termcolor import colored
import argparse

# Image argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# YOLO constants
model_weights_path = '/home/aryan-earth/capstone/yolo/helmet/helmet.weights'
model_config_path = '/home/aryan-earth/capstone/yolo/helmet/helmet.cfg'
helmet_net = cv.dnn.readNetFromDarknet(model_config_path, model_weights_path)

image = cv.imread(args['image'])

def find_helmet(image):
    model = cv.dnn_DetectionModel(helmet_net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.4)
    print(boxes)

    # Check if boxes is empty
    if (len(boxes) == 0):
        raise  Exception(colored('No helmets detected', 'red'))
    else:
        # helmet found
        helmet_detection_image = image.copy()
        no_of_helmets = len(boxes)
        helmet_count = 0
        print('No of helmets: ' + str(no_of_helmets))
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
        

            # Draw bounding box on the helmet_detection_image
            cv.rectangle(helmet_detection_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crop the image to show only the helmet
            helmet_image = image.copy()
            if (y < 0):
                y = 0
            elif (x < 0):
                x = 0
            helmet_image = helmet_image[y:y + h, x:x + w]
            # Save the ROI of the helmet
            cv.imwrite('detected_objects/{}.jpg'.format(helmet_count), helmet_image)
            helmet_count += 1
        
        # return helmet_image, helmet_detection_image
        return helmet_detection_image
        

helmet_detection_image = find_helmet(image)
cv.imwrite('helmet_detection_image.jpg', helmet_detection_image)