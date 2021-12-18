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
# Helmet
helmet_weights_path = 'helmet.weights'
helmet_config_path = 'helmet.cfg'
helmet_net = cv.dnn.readNetFromDarknet(helmet_config_path, helmet_weights_path)
# License Plate
license_plate_weights_path = "/home/glitch101/capstone_project/license_plate_weights/lp.weights"
license_plate_config_path = "/home/glitch101/capstone_project/license_plate_weights/lp.cfg"
license_plate_net = cv.dnn.readNetFromDarknet(license_plate_config_path, license_plate_weights_path)

def find_license_plate(image):
    (H, W) = image.shape[:2]

    layer_names = license_plate_net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in license_plate_net.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
        license_plate_detection_image = image.copy()
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (center_x, center_y) = (boxes[i][4], boxes[i][5])

            cv.rectangle(license_plate_detection_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return license_plate_detection_image


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

def find_helmets(image):
    (H, W) = image.shape[:2]

    layer_names = helmet_net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in helmet_net.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    helmet_net.setInput(blob)
    layer_outputs = helmet_net.forward(layer_names)

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

    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3) # Last two args are confidence and threshold respectively. 
    
    # Check if idxs is empty
    if (len(idxs) == 0):
        raise  Exception(colored('No helmets detected', 'red'))
    else:
        # helmet found
        helmet_detection_image = image.copy()
        no_of_helmets = len(idxs.flatten())
        helmet_count = 0
        print('No of helmets: ' + str(no_of_helmets))
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (center_x, center_y) = (boxes[i][4], boxes[i][5])

            # Draw bounding box on the helmet_detection_image
            cv.rectangle(helmet_detection_image, (x, y), (x + w, y + h), (255, 0, 255), 1)
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


def get_top_four_vehicles(vehicles_positions, image):
    # Get a collage of the 5 vehicles nearest to the zebra crossing
    top_four_vehicles = sorted(vehicles_positions, key=lambda x:x[1], reverse=True)
    top_four_vehicles = top_four_vehicles[:4]
    vehicles_images = []
    for vehicle in top_four_vehicles:
        x, y, w, h = vehicle[0], vehicle[1], vehicle[2], vehicle[3]
        if (y < 0):
            y = 0
        elif (x < 0):
            x = 0
        vehicle_image = image[y:y + h, x:x + w]
        vehicles_images.append(vehicle_image)
    
    # Resize images and create collage
    #todo uncomment this later
    # vehicle_0 = cv.resize(vehicles_images[0], (500,500))
    # vehicle_1 = cv.resize(vehicles_images[1], (500,500))
    # vehicle_2 = cv.resize(vehicles_images[2], (500,500))
    # vehicle_3 = cv.resize(vehicles_images[3], (500,500))

    vehicle_0 = vehicles_images[0]
    cv.imwrite("/home/glitch101/capstone_project/results/vehicle_0.jpg", vehicle_0)
    vehicle_1 = vehicles_images[1]
    cv.imwrite("/home/glitch101/capstone_project/results/vehicle_1.jpg", vehicle_1)
    vehicle_2 = vehicles_images[2]
    cv.imwrite("/home/glitch101/capstone_project/results/vehicle_2.jpg", vehicle_2)
    vehicle_3 = vehicles_images[3]
    cv.imwrite("/home/glitch101/capstone_project/results/vehicle_3.jpg", vehicle_3)
    

    horizontal_zero = np.hstack([vehicle_0, vehicle_1])
    horizontal_one = np.hstack([vehicle_2, vehicle_3])

    top_four_vehicles_image = np.vstack([horizontal_zero, horizontal_one])
    return top_four_vehicles_image

image = cv.imread(args["image"])

# # Reduce resolution of image
# scale_percent = 20 # percent of original size
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)

# resize image
# image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

# Draw threshold line
# line_image = image.copy()
# line_image = cv.line(line_image, start_point, end_point, [0, 255, 0], 2)

# Find vehicles in the image
try:
    vehicle_detection_image, vehicles_positions, vehicle_count= find_vehicles(image)
except Exception as exc:
    vehicle_detection_image = image
try:
    helmet_detection_image = find_helmets(image)
    # helmet_detection_image = cv.line(helmet_detection_image, start_point, end_point, [0, 255, 0], 1)
except Exception as exc:
    helmet_detection_image = image
    # helmet_detection_image = cv.line(helmet_detection_image, start_point, end_point, [0, 255, 0], 1)
#todo uncomment this later
try:
    top_four_vehicles_image = get_top_four_vehicles(vehicles_positions, image)
except Exception as exc:
    # top_four_vehicles_image = cv.resize(image, (1000,1000))
    top_four_vehicles_image = image

try:
    license_plate_detection_image = find_license_plate(top_four_vehicles_image)
except Exception as exc:
    license_plate_detection_image = image
    # license_plate_detection_image = image
    # license_plate_detection_image = cv.resize(image, (1000, 1000))

# image = cv.resize(image, (1000,1000))
# vehicle_detection_image = cv.resize(vehicle_detection_image, (1000,1000))
cv.imwrite("/home/glitch101/capstone_project/license_plate_results/vehicle_detection_image.jpg", vehicle_detection_image)
# helmet_detection_image = cv.resize(helmet_detection_image, (1000,1000))
cv.imwrite("/home/glitch101/capstone_project/license_plate_results/helmet_detection_image.jpg", helmet_detection_image)
# license_plate_detection_image = cv.resize(license_plate_detection_image, (1000, 1000))
cv.imwrite("/home/glitch101/capstone_project/license_plate_results/license_plate_detection_image.jpg", license_plate_detection_image)
# Write no of vehicles on vehicle_detection_image
vehicle_detection_image = cv.putText(vehicle_detection_image, str(vehicle_count), (900, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
# result = np.hstack([top_four_vehicles_image, vehicle_detection_image, helmet_detection_image])
# result = np.hstack([license_plate_detection_image, vehicle_detection_image, helmet_detection_image])# Reduce resolution of image

# Save image
# cv.imwrite("/home/glitch101/capstone_project/license_plate_results/result.jpg", result)
print("Done")