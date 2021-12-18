# Find vehicles in an image and return cropped out image of every vehicle detected in that image.

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
model_weights_path = 'yolov4.weights'
model_config_path = '/home/glitch101/darknet/cfg/yolov4.cfg'
vehicle_net = cv.dnn.readNetFromDarknet(model_config_path, model_weights_path)

image = cv.imread(args['image'])

def find_vehicle(image):
    (H, W) = image.shape[:2]

    layer_names = vehicle_net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in vehicle_net.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    vehicle_net.setInput(blob)
    layer_outputs = vehicle_net.forward(layer_names)

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
        raise  Exception(colored('No vehicles detected', 'red'))
    else:
        # Vehicle found
        vehicle_detection_image = image.copy()
        no_of_vehicles = len(idxs.flatten())
        vehicle_count = 0
        print('No of vehicles: ' + str(no_of_vehicles))
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (center_x, center_y) = (boxes[i][4], boxes[i][5])

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
        
        # return vehicle_image, vehicle_detection_image
        return vehicle_detection_image



vehicle_detection_image = find_vehicle(image)
cv.imwrite('vehicle_detection_image.jpg', vehicle_detection_image)