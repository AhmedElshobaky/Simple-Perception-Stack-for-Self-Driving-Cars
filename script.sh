#!/bin/sh
echo "Simple-Perception-Stack-for-Self-Driving-Cars"
while true
do	
		echo "Select type of input:"
		echo "	1 for image, 2 for video"	
		read input1
		if [ "$input1" == "1" ]; then
			py "YOLO_object_detection.py"
		elif [ "$input1" == "2" ]; then
			py "YOLO_object_detection_video.py"
		else
			break
		fi
done


