import numpy as np 
from PIL import Image


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

from PIL import Image, ImageDraw

def overlay_masks(mask1, mask2, color1, color2):
    
    # Ensure both masks have the same size
    if mask1.size != mask2.size:
        raise ValueError("Masks must have the same size")

    # Create a new blank image
    width, height = mask1.size
    result = Image.new('RGB', (width, height), (0, 0, 0))

    # Convert binary masks to RGB format
    mask1_rgb = mask1.convert('RGB')
    mask2_rgb = mask2.convert('RGB')

    # Create a drawing object for the resulting image
    draw = ImageDraw.Draw(result)

    # Overlay the masks with assigned colors
    for x in range(width):
        for y in range(height):
            pixel1 = mask1_rgb.getpixel((x, y))
            pixel2 = mask2_rgb.getpixel((x, y))
            if pixel1 == (255, 255, 255):
                draw.point((x, y), color1)
            if pixel2 == (255, 255, 255):
                draw.point((x, y), color2)

    return result


##------------random testing things---------------

#img = Image.open('/home/lqmeyers/paintDetect/data/full_masks/f3.1x2022_06_22.mp4.track000093.frame007395.pred.jpg')
#img = np.array(img)
#print(type(img))