# Find vehicles in an image and return cropped out image of every vehicle detected in that image.

# Imports
import numpy as np
import cv2 as cv
from termcolor import colored
import argparse

# Image argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# YOLO constants
model_weights_path = '/home/aryan-earth/capstone/yolo/vehicles/vehicles_yolov4.weights'
model_config_path = '/home/aryan-earth/capstone/yolo/vehicles/vehicles_yolov4.cfg'
vehicle_net = cv.dnn.readNetFromDarknet(model_config_path, model_weights_path)


def find_vehicle(image):

    model = cv.dnn_DetectionModel(vehicle_net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.4)
    print(classIds)

    # Check if boxes is empty
    if (len(boxes) == 0):
        raise  Exception(colored('No vehicles detected', 'red'))
    else:
        # Vehicle found
        vehicle_detection_image = image.copy()
        no_of_vehicles = len(boxes)
        vehicle_count = 0
        print('No of vehicles: ' + str(no_of_vehicles))

        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            # Draw bounding box on the vehicle_detection_image
            cv.rectangle(vehicle_detection_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Crop the image to show only the vehicle
            vehicle_image = image.copy()
            if (y < 0):
                y = 0
            elif (x < 0):
                x = 0
            vehicle_image = vehicle_image[y:y + h, x:x + w]
            # Save the ROI of the vehicle
            cv.imwrite('detected_objects/{}.jpg'.format(vehicle_count), vehicle_image)
            vehicle_count += 1

        return vehicle_detection_image



image = cv.imread(args['image'])
vehicle_detection_image = find_vehicle(image)
cv.imwrite('vehicle_detection_image.jpg', vehicle_detection_image)
