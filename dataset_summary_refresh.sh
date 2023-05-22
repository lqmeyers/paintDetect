#!/bin/bash/

#a program to automatically print the contents of the dataset folders to a text file to record dataset distribution
CURRENTDATE=`date +"%Y-%m-%d-%T"`
echo "Dataset summary last refreshed: "$CURRENTDATE > dataset_summary.txt
echo " " >> dataset_summary.txt
echo "current dir is " >> dataset_summary.txt
pwd >> dataset_summary.txt
echo " " >> dataset_summary.txt
echo "-------------------------------------------------------------------" >> dataset_summary.txt
echo "current files in images/" >> dataset_summary.txt
echo " " >> dataset_summary.txt
ls -R ./images/ >> dataset_summary.txt
echo " " >> dataset_summary.txt
echo "-------------------------------------------------------------------" >> dataset_summary.txt
echo "current files in masks/" >> dataset_summary.txt
echo " " >> dataset_summary.txt
ls -R ./masks/ >> dataset_summary.txt
