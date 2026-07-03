#a file for using unet segmentation to do psuedo pose estimation and get a vector that 
#describes bees relative position, and can be used for telling right from left in paint codes


import torch
import os 
import numpy as np
import torch
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
import sys
import cv2
from segmentation import Segmentation
from current_models import trained_models

def overlay_masks(mask1, mask2):
    
    # Ensure both masks have the same size
    if mask1.size != mask2.size:
        raise ValueError("Masks must have the same size")

    # Create a new blank image
    width, height = mask1.size
    result = Image.new('L', (width, height), 0)


    # Convert binary masks to RGB format
    #mask1_rgb = mask1.convert('RGB')
    #mask2_rgb = mask2.convert('RGB')

    # Create a drawing object for the resulting image
    draw = ImageDraw.Draw(result)

    # Overlay the masks with assigned colors
    for x in range(width):
        for y in range(height):
            pixel1 = mask1.getpixel((x, y))
            pixel2 = mask2.getpixel((x, y))
            if pixel1 == 255:
                draw.point((x, y), 255)
            if pixel2 == 255:
                draw.point((x, y), 200)

    return result

def get_rot_rect(input):
# Convert to numpy 
    overlay_result = np.array(input)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(overlay_result, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        combined_contour = np.concatenate(contours)
    else:
        combined_contour = contours[0]

    # Fit a rotated rectangle to the contour
    rotated_rect = cv2.minAreaRect(combined_contour)
    return(rotated_rect)

def get_centerline_endpoints(center1, center2, rotated_rect):
    center1 = np.array(center1)
    center2 = np.array(center2)
    
    centerline_direction = center2 - center1
    centerline_direction /= np.linalg.norm(centerline_direction)

    box = cv2.boxPoints(rotated_rect)
    rect_side_vectors = [box[i] - box[(i + 1) % 4] for i in range(4)]
    rect_side_lengths = [np.linalg.norm(side_vector) for side_vector in rect_side_vectors]
    longest_side_idx = np.argmax(rect_side_lengths)

    side_vector1 = box[longest_side_idx] - box[(longest_side_idx + 1) % 4]
    side_vector2 = box[(longest_side_idx + 2) % 4] - box[(longest_side_idx + 3) % 4]

    if np.dot(centerline_direction, side_vector1) > np.dot(centerline_direction, side_vector2):
        centerline_angle = np.arctan2(side_vector1[1], side_vector1[0])
    else:
        centerline_angle = np.arctan2(side_vector2[1], side_vector2[0])

    half_length = 0.5 * rect_side_lengths[longest_side_idx]
    endpoint1 = rotated_rect[0] + half_length * np.array([np.cos(centerline_angle), np.sin(centerline_angle)])
    endpoint2 = rotated_rect[0] - half_length * np.array([np.cos(centerline_angle), np.sin(centerline_angle)])

    return endpoint1, endpoint2

"""
def get_centerline_endpoints(rotated):
    center, (width, height), angle_degrees = rotated
    angle_radians = np.radians(angle_degrees)
    half_longer_side = max(width, height) / 2.0

    endpoint1 = (
        center[0] + half_longer_side * np.cos(angle_radians),
        center[1] + half_longer_side * np.sin(angle_radians)
    )

    endpoint2 = (
        center[0] - half_longer_side * np.cos(angle_radians),
        center[1] - half_longer_side * np.sin(angle_radians)
    )

    return endpoint1, endpoint2
"""

import math
def pythagMe(coord1,coord2):
    '''performs pythagorean theorum to return the
    distance between two points. Input coords as [x,y]'''
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    return math.sqrt((y2-y1)**2+(x2-x1)**2)

def order_endpoints(h_center,endpoints):
    '''returns endpoints in order of first point is closest to center of head'''
    dist1 = pythagMe(h_center,endpoints[0])
    dist2 = pythagMe(h_center,endpoints[1])
    if dist1 >= dist2:
        return((endpoints[1],endpoints[0]))
    else:
        return(endpoints)


def get_angle(image_path,save_vis=False,outpath='./'):
    '''
        Final function to do pseudo pose estimation from UNEt segmentations of head and abdomen.
        Takes image, predicts location of head and thorax and returns a line with the first endpoint closest 
        to the head, parrelel with a bounding box around head and thorax, roughly equal to a skeleton segment 
        through the head and thorax. Has options to save a visualization image as well.
        
        params:
        image_path(str): path to input image 

        returns:
        endpoints (tuple): tuple of coordinates of endpoints of centerline, with head side line first

    '''
    #load image
    img = Image.open(image_path)
    
    #segment image and load thorax and head masks
    seg = Segmentation(trained_models,img)
    head = seg.head 
    thorax = seg.thorax

    # Overlay masks and save the result
    overlay_result = overlay_masks(head, thorax)

    rotated_rect = get_rot_rect(overlay_result)
    h_rect = get_rot_rect(head)
    t_rect = get_rot_rect(thorax)

    h_cent = h_rect[0]
    t_cent = t_rect[0]
   
    pose_line = get_centerline_endpoints(h_cent,t_cent,rotated_rect)
    pose_line = order_endpoints(h_cent,pose_line)

    #convert to numpy for vis
    overlay_result = np.array(overlay_result)
    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    if save_vis == True:
        cv2.line(img, np.int0(pose_line[0]), np.int0(pose_line[1]), (0, 0, 255), 2)
        kp = [cv2.KeyPoint(x = int(pose_line[0][0]) ,y = int(pose_line[0][1]), size=4)]
        cv2.drawKeypoints(img,kp,img,(0,255, 0))
        cv2.imwrite(outpath+os.path.basename(image_path)[:-4]+'.line_vis.png',img)
    return pose_line

"""
    # Draw the rotated rectangle on the original image
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)

    print(box)
    pose_line = get_centerline_endpoints(rotated_rect)

    print(pose_line)

    #cv2.drawContours(overlay_result, [box], 0, (255, 255, 255), 2)
    cv2.line(overlay_result, np.int0(pose_line[0]), np.int0(pose_line[1]), (0, 255, 0), 2)
"""

#test_image ='/home/lqmeyers/paintDetect/data/images/testing/f1.2x2022_06_22.mp4.track000004.frame001173.jpg'
#test_image = sys.argv[0] 
#out_path = sys.argv[1]
#get_angle(test_image,True,out_path)

image_dir = '/home/agomez/batch_1/young-adults-blue-white-in-lab-1-32_batch_1/'

dir_list = os.walk(image_dir)
for root, dirs, files in dir_list:
    files.sort()
    for i in range(len(files)):
        f = files[i]
        if i % 100 == 0 and f[-4:] in ['.jpg','.png']:
            get_angle(root+r'/'+f,True,'/home/lmeyers/paintDetect/data/images/lines/andrea_test/')


