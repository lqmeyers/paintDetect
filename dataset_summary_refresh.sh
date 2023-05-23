#!/bin/bash/

#a program to automatically print the contents of the dataset folders to a text file to record dataset distribution
CURRENTDATE=`date +"%Y-%m-%d-%T"`
echo "Dataset summary last refreshed: "$CURRENTDATE > dataset_summary.txt
echo " " >> dataset_summary.txt
echo "-------------------------------------------------------------------" >> dataset_summary.txt
echo "current datasets/" >> dataset_summary.txt
echo " " >> dataset_summary.txt
ls -R ./data/ >> dataset_summary.txt
