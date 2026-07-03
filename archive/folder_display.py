
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_in_folder(folder_path, num_cols=10):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Calculate the number of rows needed based on the number of columns specified
    num_rows = (len(image_files) - 1) // num_cols + 1
    
    # Create a new figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    
    # Loop through the image files and display each image
    for i, image_file in enumerate(image_files):
        row = i // num_cols
        col = i % num_cols
        
        image_path = os.path.join(folder_path, image_file)
        img = mpimg.imread(image_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    # Remove any empty subplots
    for i in range(len(image_files), num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
folder_path = '/home/lqmeyers/paintDetect/data/images/lines/'
display_images_in_folder(folder_path)
