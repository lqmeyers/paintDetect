#!/bin/bash/
#f22x2022_06_28.mp4.track000004.frame000372.jpg
pwd
CURRENTDATE=`date +"%Y-%m-%d-%T"`
mkdir /home/lmeyers/paintDetect/masks/predict_$CURRENTDATE

for f in /home/lmeyers/paintDetect/images/training/*; do
	echo python ./Pytorch-UNet/predict.py -i $f -m /home/lmeyers/paintDetect/wandb/run-20230517_224404-8xbgtqw6/files/model.pth -o /home/lmeyers/paintDetect/masks/predict_$CURRENTDATE${f:41:-4}".pred.jpg"
done
