# Imports
import numpy as np
from cv2 import cv2 as cv
from numpy.core.fromnumeric import resize, sort
from termcolor import colored
import argparse
from time import sleep

# frame argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# YOLO constants
# Vehicle
vehicle_weights_path = 'yolov4.weights'
vehicle_config_path = '/home/glitch101/darknet/cfg/yolov4.cfg'
vehicle_net = cv.dnn.readNetFromDarknet(vehicle_config_path, vehicle_weights_path)

def find_vehicles(image):
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
        return -1
    else:
        # Vehicle found
        vehicle_detection_image = image.copy()
        no_of_vehicles = len(idxs.flatten())
        vehicles_positions = []
        vehicle_count = 0
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (center_x, center_y) = (boxes[i][4], boxes[i][5])

            vehicle_outmost_point = (x+w, y+h)
            cv.rectangle(vehicle_detection_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            if (y < 0):
                y = 0
            elif (x < 0):
                x = 0
            # Save the ROI of the vehicle
            vehicles_positions.append([x, y, w, h])
            vehicle_count += 1

        print('No of vehicles: ' + str(vehicle_count))
        # return vehicle_image, vehicle_detection_image
        return vehicle_detection_image, vehicles_positions, vehicle_count


image = cv.imread(args["image"])

# Find vehicles in the image
try:
    vehicle_detection_image, vehicles_positions, vehicle_count= find_vehicles(image)
except Exception as exc:
    print('No vehicles found!')


# Extract vehicles images
vehicle_count = 0
for vehicle_position in vehicles_positions:
    try:
        x, y, w, h = vehicle_position[0], vehicle_position[1], vehicle_position[2], vehicle_position[3]
        vehicle_image = image[y:y+h, x:x+w]
        cv.imwrite('/home/glitch101/capstone_project/results/vehicle_{}.jpg'.format(vehicle_count), vehicle_image)
        vehicle_count += 1
    except:
        print('Failed to extract vehicle image')

cv.imwrite("/home/glitch101/capstone_project/results/vehicle_detection_image.jpg", vehicle_detection_image)
print("Done")