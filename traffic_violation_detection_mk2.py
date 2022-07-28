# Imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cv2 import cv2 as cv
import pytesseract
from numpy.core.fromnumeric import resize, sort
from termcolor import colored
import argparse
from time import sleep
from exceptions import NoObjectsFoundException, NoVehiclesFoundException, NotWearingHelmetException, CantFindLicensePlateException, NoPersonFoundException

# frame argument
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to input video')
args = vars(ap.parse_args())

# YOLO constants
# Vehicles
objects_weights_path = '/home/aryan-earth/capstone/yolo/vehicles/vehicles_yolov4.weights'
objects_config_path = '/home/aryan-earth/capstone/yolo/vehicles/vehicles_yolov4.cfg'
objects_net = cv.dnn.readNetFromDarknet(objects_config_path, objects_weights_path)
# Helmet
helmet_weights_path = '/home/aryan-earth/capstone/yolo/helmet/helmet.weights'
helmet_config_path = '/home/aryan-earth/capstone/yolo/helmet/helmet.cfg'
helmet_net = cv.dnn.readNetFromDarknet(helmet_config_path, helmet_weights_path)
# License Plate
license_plate_weights_path = "/home/aryan-earth/capstone/yolo/license_plate/license_plate.weights"
license_plate_config_path = "/home/aryan-earth/capstone/yolo/license_plate/license_plate.cfg"
license_plate_net = cv.dnn.readNetFromDarknet(license_plate_config_path, license_plate_weights_path)


PERSON = 0
CYCLE = 1
CAR = 2
TWO_WHEELER = 3
BUS = 5
TRUCK = 7
TRAFFIC_LIGHT = 9

objects_ids_dict = {0: 'PERSON', 1: 'CYCLE', 2: 'CAR', 3: "TWO_WHEELER",
                            5: 'BUS', 7: 'TRUCK', 9: 'TRAFFIC_LIGHT'}
                            # yellow, pink, green, blue, orange, brown, white
                            #note truck class also contains autorickshaws and other commercial vehicles.
objects_bb_color_dict = {0: (0, 255, 255), 1: (171, 64, 255), 2: (0, 255, 198), 3: (255, 138, 68),
                            5: (0, 152, 255), 7: (72, 85, 121), 9: (245, 245, 245)}
vehicles_ids = [2, 3, 5, 7]

# video 1
# threshold_line_coordinates = [(0, 950), (1920, 500)]  # [start. end]
# video 2
threshold_line_coordinates = [(0, 950), (1920, 950)]  # [start. end]

license_plates = list()



def draw_threshold_line(frame):
    threshold_line_image = frame.copy()
    threshold_line_image = cv.line(threshold_line_image, threshold_line_coordinates[0], threshold_line_coordinates[1], [0, 255, 0], 2)
    return threshold_line_image

def extract_objects(frame):

    objects_detection_image = frame.copy()

    model = cv.dnn_DetectionModel(objects_net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)

    # Check if boxes is empty
    if (len(boxes) == 0):
        raise NoObjectsFoundException
    else:
        # Objects present
        detection_image = frame.copy()
        count = 0
        objects_data = list() # used to store the objects type (classId) and position.

        for obj_no in range(len(boxes)):
            # extract the bounding box coordinates
            (x, y) = (boxes[obj_no][0], boxes[obj_no][1])
            (w, h) = (boxes[obj_no][2], boxes[obj_no][3])
            obj_center = [int(x + w/2), int(y + h/2)] # [x, y]
            object_id = classIds[obj_no]
            

            if (object_id in objects_ids_dict):
                object_name = objects_ids_dict.get(object_id)
                bb_color = objects_bb_color_dict.get(object_id)
                objects_position = [x, y, w, h]
                # Draw the bounding box
                cv.rectangle(objects_detection_image, (x, y), (x + w, y + h), bb_color, 2)
                cv.imwrite('dashboard/objects_detection_image.png', objects_detection_image)
                # Write the object name
                cv.putText(objects_detection_image, object_name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, bb_color, 2)

                # Save the objects position
                objects_data.append([object_id, objects_position, obj_center])


                count += 1

        return objects_detection_image, objects_data, count


def extract_vehicles(frame, objects_data):
    # Objects with ids {2, 3, 5, 7, 9} are vehicles
    vehicles_data = list()
    no_of_vehicles = 0
    vehicles_detection_image = frame.copy()
    vehicles_found_ids = list()
    # Find the vehicles
    for obj in objects_data:
        object_id = obj[0]
        if (object_id in vehicles_ids):
            # Vehicle found
            bb_color = objects_bb_color_dict.get(object_id)
            # Save the vehicle position
            vehicles_data.append([obj[0], obj[1], obj[2]])
            x, y = obj[1][0], obj[1][1]
            w, h = obj[1][2], obj[1][3]
            
            cv.rectangle(vehicles_detection_image, (x, y), (x + w, y + h), bb_color, 2)
            cv.imwrite('dashboard/vehicles_detection_image.png', vehicles_detection_image)

            no_of_vehicles += 1
            vehicles_found_ids.append(object_id)

    if (no_of_vehicles != 0):
        # Find the no of vehicles of each type
        vehicles_stats = np.unique(vehicles_found_ids, return_counts=True)
        return vehicles_detection_image, vehicles_data, no_of_vehicles, vehicles_stats
    else:
        raise NoVehiclesFoundException


def extract_persons(frame, objects_data):
    persons_data = list()
    no_of_persons = 0
    persons_detection_image = frame.copy()
    # Extract the persons coordinates
    for obj in objects_data:
        if (obj[0] == PERSON):
            
            # A person found
            bb_color = objects_bb_color_dict.get(PERSON)
            position = obj[1]
            center = obj[2]
            # Save the vehicle position
            persons_data.append([position, center])
            x, y = position[0], position[1]
            w, h = position[2], position[3]
            
            cv.rectangle(persons_detection_image, (x, y), (x + w, y + h), bb_color, 2)
            cv.imwrite('dashboard/persons_detection_image.png', persons_detection_image)

            no_of_persons += 1

    if (no_of_persons != 0):
        return persons_detection_image, persons_data, no_of_persons
        
    else:
        raise NoPersonFoundException
    


def print_vehicle_stats(vehicles_stats):
    unique_vehicles_ids, unique_vehicles_count = vehicles_stats
    for i in range(len(unique_vehicles_ids)):
        veh_id = unique_vehicles_ids[i]
        vehicle_name = objects_ids_dict[veh_id]
        vehicle_count = unique_vehicles_count[i]
        print('[{}] [{}]'.format(vehicle_name, vehicle_count))

    return vehicles_stats

def find_helmet(human_image):
    model = cv.dnn_DetectionModel(helmet_net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(human_image, confThreshold=0.6, nmsThreshold=0.4)

    # Check if boxes is empty
    if (len(boxes) == 0):
        raise NotWearingHelmetException
    else:
        # Helmet found
        no_of_helmets = len(boxes)
        print('Helmet found: ', no_of_helmets)
        helmet_detection_image = human_image.copy()
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            # Draw bounding box on the helmet
            cv.rectangle(helmet_detection_image, (x, y), (x + w, y + h), (252, 128, 234), 1)
            return helmet_detection_image
            


def bulk_helmet_violation_check(frame, objects_data):
    
    humans_tw_centers_list = list()
    humans_position_list = list()
    two_wheelers_centers_list = list()
    humans_centers_list = list()
    closest_points_pairs_list = list()

    humans_tw_centers_image = frame.copy()
    for obj in objects_data:
        obj_id = obj[0]
        if (obj_id in [0, 3]):
            # A human or two wheeler found
            obj_position = obj[1]
            x, y = obj_position[0], obj_position[1]
            w, h = obj_position[2], obj_position[3]
            obj_center = [int(x + w/2), int(y + h/2)] # [x, y]
            humans_tw_centers_list.append(obj_center)
            if (obj_id == 3):
                # A two wheeler
                two_wheelers_centers_list.append(obj_center)
            elif (obj_id == 0):
                humans_centers_list.append(obj_center)
                humans_position_list.append([x, y, w, h, obj_center])
            # Draw the bounding box and center
            cv.rectangle(humans_tw_centers_image, (x, y), (x + w, y + h), objects_bb_color_dict[obj_id], 2)
            cv.circle(humans_tw_centers_image, (obj_center[0], obj_center[1]) , 0, objects_bb_color_dict[obj_id], 10)
            cv.putText(humans_tw_centers_image, str((obj_center[0], obj_center[1])), (obj_center[0], obj_center[1]), cv.FONT_HERSHEY_SIMPLEX, 1, objects_bb_color_dict[obj_id], 2)


    # For every two wheeler's center, find the closest human to it. That should be the driver.
    two_wh_no = 0
    for two_wheeler_center in two_wheelers_centers_list:
        #print('Finding neighbour of two wheeler no: : ', two_wh_no)
        a = np.asarray(humans_centers_list)
        b = np.asarray([two_wheeler_center])
        c = np.append(a, b, axis=0)
        #print(c)
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(c)
        distances, indices = nbrs.kneighbors(c)
        #print(indices)
        #print(distances)

        closest_point_index = indices[-1][1]
        # Coordinates of the closest point
        closest_point = c[closest_point_index]
        #print('The point closest to {} is {}'.format(two_wheeler_center, closest_point))

        closest_points_pairs_list.append([two_wheeler_center, closest_point])
        # Draw the line connecting the points
        cv.line(humans_tw_centers_image, two_wheeler_center, closest_point, [0, 255, 0], 2)

        human_bb_coordinates = humans_position_list[closest_point_index]
        human_image = frame[human_bb_coordinates[1]:human_bb_coordinates[1] + human_bb_coordinates[3],
                            human_bb_coordinates[0]:human_bb_coordinates[0] + human_bb_coordinates[2]]


        # Now, find helmet in the image
        try:
            # show the person in normal color bb
            helmet_detection_image = find_helmet(human_image)
            print('Human {} is wearing helmet'.format(two_wh_no))
            cv.imwrite('dashboard/helmet_detection_image_{}.png'.format(two_wh_no), helmet_detection_image)
        except Exception as exc:
            print('Human {} is NOT wearing helmet'.format(two_wh_no))
            print(exc)
            # make the bounding box red
            # Extract license plate
            
            

        

        two_wh_no += 1
        
    print('closest_points_pairs_list', closest_points_pairs_list)

    return humans_tw_centers_image

def check_helmet(frame, vehicle, persons_data):
    # Extract the image of the driver of the two wheeler and check if they are wearing a helmet.
    # We do this by finding the nearest human to the vehicle in focus
    # We calculate the center of the bounding box of the vehicle and
    # then calculate the distance between it and the centers of the bounding boxes of
    # all the humans in the frame.
    # The one closest must belong to the driver.


    x, y, w, h = vehicle[1][0], vehicle[1][1], vehicle[1][2], vehicle[1][3]
    vehicle_center= [int (x + (w/2)), int (y + (h/2))] # The center of the bounding box of the violators vehicle

    # Extract the centers of the persons
    people_centers_list = list()
    for person in persons_data:
        position = person[0]
        center = person[1]

        people_centers_list.append(center)

    # Append the vehicle_center to the end of persons_data array
    a = np.asarray(people_centers_list)
    b = np.asarray([vehicle_center])
    c = np.append(a, b, axis = 0)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(c)
    distances, indices = nbrs.kneighbors(c)


    closest_point_index = indices[-1][1]
    # Coordinates of the closest point
    closest_point = c[closest_point_index]

    


    person_coordinates = persons_data[closest_point_index]
    person_image = frame[person_coordinates[0][1]:person_coordinates[0][1] + person_coordinates[0][3],
                        person_coordinates[0][0]:person_coordinates[0][0] + person_coordinates[0][2]]


    # Now, find helmet in the image
    try:
        _ = find_helmet(person_image)
        return True, person_coordinates
    except NotWearingHelmetException as exc:
        return False, person_coordinates



def extract_license_plate(vehicle_image):
    model = cv.dnn_DetectionModel(license_plate_net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(vehicle_image, confThreshold=0.6, nmsThreshold=0.4)

    # Check if boxes is empty
    if (len(boxes) == 0):
        raise CantFindLicensePlateException
    else:
        # print(colored('License plate extracted', 'green', attrs=['bold', 'blink']))
        print('License plate extracted')
        # License plate found
        license_plate_detection_image = vehicle_image.copy()
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            # Draw bounding box on the helmet
            cv.rectangle(license_plate_detection_image, (x, y), (x + w, y + h), (252, 128, 234), 1)
            cv.imwrite('dashboard/license_plate_detection_image.png', license_plate_detection_image)
            license_plates.append(license_plate_detection_image)
            return license_plate_detection_image

def run_ocr_license_plate(license_plate_image):
    return pytesseract.image_to_string(license_plate_image)

def is_light_jumped(vehicle):
    # Calculate the bottom line mid point
    vehicle_coordinates = vehicle[1]
    x, y, w, h = vehicle_coordinates[0], vehicle_coordinates[1], vehicle_coordinates[2], vehicle_coordinates[3]
    #todo Maybe use Nearest Neighbour instead for better accuracy.
    bottom_middle_point_x, bottom_middle_point_y = int(x + w/2), int(y+ h)

    if (bottom_middle_point_y >= threshold_line_coordinates[0][1]):
        # Light Jumped.
        return True
    else:
        return False


def mark_traffic_light_violator(violator_detection_image, vehicle_coordinates):
    x, y, w, h = vehicle_coordinates[0], vehicle_coordinates[1], vehicle_coordinates[2], vehicle_coordinates[3]
    # Draw a red bounding box over the violator
    cv.rectangle(violator_detection_image, (x, y), (x + w, y + h), [0, 0, 255], 2)
    # Display a violator text over the bounidng box
    cv.putText(violator_detection_image, 'Violator', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)

    return violator_detection_image

def mark_helmet(helmet_detection_image, vehicle_coordinates, isWorn):
    x, y, w, h = vehicle_coordinates[0], vehicle_coordinates[1], vehicle_coordinates[2], vehicle_coordinates[3]
    if (isWorn):
        # Draw a red bounding box over the violator
        cv.rectangle(helmet_detection_image, (x, y), (x + w, y + h), [255, 255, 255], 2)
        # Display a violator text over the bounidng box
        cv.putText(helmet_detection_image, 'HELMET: YES', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
    else:
        cv.rectangle(helmet_detection_image, (x, y), (x + w, y + h), [0, 0, 255], 2)
        # Display a violator text over the bounidng box
        cv.putText(helmet_detection_image, 'HELMET: NO', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)

    return helmet_detection_image

video = cv.VideoCapture(args['video'])


fourcc = cv.VideoWriter_fourcc(*'MJPG')
raw_footage = cv.VideoWriter('dashboard/raw_footage.avi', fourcc, 20.0, (3840, 2160))

# Dashboard window
# cv.namedWindow('dashboard', cv.WINDOW_NORMAL)



# # slope of threshold line
# m = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

frame_no = 0
while video.isOpened():
    ret, frame = video.read()

    # Skip every 5 frames
    if (frame_no % 2 == 0):
        try:
            # Draw the threshold line
            threshold_line_image = draw_threshold_line(frame)
            # Extract objects in the frame
            objects_detection_image, objects_data, no_of_objects = extract_objects(frame)
            # Extract people data
            persons_detection_image, persons_data, no_of_people = extract_persons(frame, objects_data)
            # Extract vehicles from the objects data
            vehicles_detection_image, vehicles_data, no_of_vehicles, vehicles_stats = extract_vehicles(frame, objects_data)

            # Print the frame stats
            print('[Objects] [{}] [Persons] [{}]  [Vehicles] [{}]'.format(no_of_objects, no_of_people, no_of_vehicles))
            # Print the vehicles stats
            vehicles_stats = print_vehicle_stats(vehicles_stats)
            unique_vehicles_ids, unique_vehicles_count = vehicles_stats

            violator_detection_image = vehicles_detection_image.copy()
            helmet_detection_image = frame.copy()
            
            for vehicle in vehicles_data:
                vehicle_id = vehicle[0]
                vehicle_type = objects_ids_dict[vehicle_id]
                vehicle_coordinates = vehicle[1]
                vehicle_image = frame[vehicle_coordinates[1]:vehicle_coordinates[1] + vehicle_coordinates[3],
                                      vehicle_coordinates[0]:vehicle_coordinates[0] + vehicle_coordinates[2]]
                if (is_light_jumped(vehicle)):
                    print('============================================================')
                    # print(colored('Traffic light jumped', 'red', attrs=['bold', 'blink']))
                    print('Traffic light jumped')
                    # Find vehicle type and take appropriate actions
                    # Mark the bounding box red
                    violator_detection_image = mark_traffic_light_violator(violator_detection_image, vehicle_coordinates)
                    if (vehicle_id == 3):
                        # print(colored('Violator type: {}'.format(vehicle_type), 'cyan', attrs=['bold']))
                        print('Violator type: {}'.format(vehicle_type))
                        # A two wheeler
                        # Check if the violater is not wearing a helmet too.
                        is_wearing_helmet, person_coordinates = check_helmet(frame, vehicle, persons_data)
                        if (is_wearing_helmet):
                            # Mark the helmet
                            # print(colored('helmet: YES', 'green'))
                            print('helmet: YES')
                            violator_detection_image = mark_helmet(helmet_detection_image, person_coordinates[0], isWorn=True)
                        else :
                            # Write HELMET:NO over the right side of the bounding box
                            # print(colored('helmet: NO', 'red'))
                            print('helmet: NO')
                            violator_detection_image = mark_helmet(helmet_detection_image, person_coordinates[0], isWorn=False)

                        license_plate_image = extract_license_plate(vehicle_image)
                        # vehicle_number = run_ocr_license_plate(license_plate_image)
                        # print(colored('Vehicle Number: {}'.format(vehicle_number), 'green', attrs=['bold']))

                    else:
                        # print(colored('Violator type: {}'.format(vehicle_type), 'yellow', attrs=['bold']))
                        print('Violator type: {}'.format(vehicle_type))
                        # Any other vehicle
                        # Try to extract license plate
                        license_plate_image = extract_license_plate(vehicle_image)
                        # vehicle_number = run_ocr_license_plate(license_plate_image)
                        # print(colored('Vehicle Number: {}'.format(vehicle_number), 'green', attrs=['bold']))
                else:
                    # Do nothing
                    pass
        

        except NoObjectsFoundException as exc:
            # print(colored('No objects found', 'red', attrs=['bold']))
            print('No objects found')
            objects_detection_image  = frame
            print('============================================================')

        except NoPersonFoundException as exc:
            # print(colored('No person found', 'red', attrs=['bold']))
            print('No person found')
            persons_detection_image  = frame
            print('============================================================')

        except NoVehiclesFoundException as exc:
            # print(colored('No vehicles found', 'red', attrs=['bold']))
            print('No vehicles found')
            vehicles_detection_image  = frame
            print('============================================================')

        except NotWearingHelmetException as exc:
            # print(colored('Not wearing helmet', 'red', attrs=['bold', 'blink']))
            print('Not wearing helmet')
            print('============================================================')

        except CantFindLicensePlateException as exc:
            # print(colored('Cannot find license plate', 'red', attrs=['bold']))
            print('Cannot find license plate')
            print('============================================================')

        except Exception as exc:
            # print(colored(exc, 'red'))
            print(exc)
            print('============================================================')


        row_0 = np.hstack([threshold_line_image, objects_detection_image])
        row_1 = np.hstack([violator_detection_image, persons_detection_image])
        result = np.vstack([row_0, row_1])
        cv.imwrite('dashboard/result.png', result)
        try:
            raw_footage.write(result)
        except Exception as exc:
            print(exc)
            raw_footage.release()            

        # print(colored('Frame number {} complete'.format(frame_no), 'green'))
        print('Frame number {} complete'.format(frame_no))

        frame_no += 1

    else:
        frame_no += 1


video.release()
raw_footage.release()