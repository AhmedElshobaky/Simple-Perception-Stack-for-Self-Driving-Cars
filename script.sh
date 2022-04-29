#!/bin/sh
echo "Simple-Perception-Stack-for-Self-Driving-Cars"
while true
do
	echo "What type of input: -image -video -q"
	read input
	if  [ "$input" = "image" ]; then
		python "lane.py"
	  elif [ "$input" = "video" ]; then
		python "lane_video.py"
	  else
		break
	fi
done


