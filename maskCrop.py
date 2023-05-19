##Takes an image and a binary mask and saves a new image cropped to where the mask was 1


#I really should start making my code cleaner
#going to try argparse
import argparse
import cv2
import matplotlib 
import matplotlib.pyplot as plt 


#----------------global inputs-----------------
"""
def main(args):
    # Your script logic here
    print(args)

def get_args():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--input', '--image', '-i', metavar='INPUT', nargs='+', help='full path and name of input images', required=True)
    parser.add_argument('--mask', '-m', metavar='MASK', nargs='+',help= 'full path and name of mask file',required=True)
    # Add more arguments as needed
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
"""
##---------file handling utils---------------------
def getPath(file):
    '''uses a path string to get just the directoy of a file'''
    strOut = ''
    i = 1
    while file[-i] != '/':
        i = i + 1
        #print(file[-i])
    strOut = file[0:len(file)-(i-1)]
    return strOut

def getName(file):
    '''uses a path string to get the name of a file'''
    strOut = ''
    i = 1
    while file[-i] != '/':
        i = i + 1
        #print(file[-i])
    strOut = file[-(i-1):]
    return strOut

#---------img operations-------------------------------

def crop_by_mask(image,mask):
    """
    takes in an image, and a corresponding binary mask and saves new file of image 
    values only on pixels where mask was positive. 

    Parameters:
    image (str): a string of path to image.
    mask (str): a string og path to mask 

    Returns:
    saves file to image_path/image_name.paint_only.png
    """
    img = cv2.imread(image)
    mask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
    output_path = getPath(image)
    image_name = getName(image)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found in the mask.")
        return

    # Find the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image based on the bounding box
    cropped_image = img[y:y+h, x:x+w]

    # Create a mask of the cropped region
    cropped_mask = mask[y:y+h, x:x+w]

    # Apply the mask to the cropped image
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    # Save the result to the output path
    #cv2.imwrite(output_path+image_name+".paint_masked.png", result)
    cv2.imshow('crop',result)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    print(f"Cropped image saved to: {output_path+image_name}.paint_masked.png")

# Example usage
image_path = "/home/lqmeyers/paintDetect/images/testing/f17x2022_06_28.mp4.track000206.frame006589.jpg"
mask_path = "/home/lqmeyers/paintDetect/masks/testing/f17x2022_06_28.mp4.track000206.frame006589.Paint.png"
crop_by_mask(image_path,mask_path)


