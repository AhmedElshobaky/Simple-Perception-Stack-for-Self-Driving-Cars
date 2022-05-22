#!/bin/sh
echo "Simple-Perception-Stack-for-Self-Driving-Cars"
while true
do
	echo "Select mode:"
	echo "	1 for only lane detection"
	echo "	2 for Lane and cars detection"
	read input1
	echo "Select type of input:"
	echo "	1 for image, 2 for video"	
	read input2
	if [ "$input1" == "1" ]; then
		cd part1			
		if [ "$input2" == "1" ]; then
			py "lane.py"
		elif [ "$input2" == "2" ]; then
			py "lane_video.py"
		else
			break
		fi
	
	elif [ "$input1" == "2" ]; then			
		cd part2
		if [ "$input2" == "1" ]; then
			py "YOLO_object_detection.py"
		elif [ "$input2" == "2" ]; then
			py "YOLO_object_detection_video.py"
		else
			break
		fi
	else
		break
	fi
	cd ../	
done

