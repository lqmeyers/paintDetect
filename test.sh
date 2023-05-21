#!/bin/bash
 
CURRENTDATE=`date +"%Y-%m-%d %T"`
CURRENTDATEONLY=`date +"%b %d, %Y"`
CURRENTEPOCTIME=`date +"%Y-%m-%d %T"`
 
echo Current Date is: ${CURRENTDATEONLY}
echo Current Date and Time is: `date +"%Y-%m-%d %T"`
echo Current Date and Time is: ${CURRENTDATE}
echo Current Unix epoch time is: ${CURRENTEPOCTIME}
