import fiftyone as fo

#name for dataset
name = "full masks pred 6/13"

#dir containing dataset
dir = "/home/lqmeyers/paintDetect/data/full_masks/predict_2023-06-13-21:09:35/"

dataset = fo.Dataset.from_images_dir(dir)

session = fo.launch_app(dataset,port=5151)
