import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_and_get_hsv_value(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB for display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure and show the image
    plt.figure()
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Click on a pixel to see its HSV value")
    plt.show()

    # Wait for the user to click on a pixel
    click_event = plt.ginput(1)

    # Get the coordinates of the clicked pixel
    x, y = int(click_event[0][0]), int(click_event[0][1])

    # Get the RGB value of the clicked pixel
    rgb_value = image[y, x]

    # Convert the RGB value to HSV
    hsv_value = cv2.cvtColor(np.uint8([[rgb_value]]), cv2.COLOR_RGB2HSV)[0][0]

    return hsv_value

# Example usage:
image_path = '/home/lqmeyers/SLEAP_files/Bee_imgs/flowerpatch_imgs/filesort_CVAT_sample/f2x2022_06_22.mp4.track000022.frame004129.jpg'
hsv_value = show_image_and_get_hsv_value(image_path)
print("HSV Value:", hsv_value)