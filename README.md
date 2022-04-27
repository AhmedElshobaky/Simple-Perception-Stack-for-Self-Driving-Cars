# Simple-Perception-Stack-for-Self-Driving-Cars
In this project we are going to create a simple perception stack for self-driving cars (SDCs.). Our only data source will be video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks.

## How to run the code:
### Lane notebook (input: image):
 - Download lane.py file then open cmd in the directory of the lane.py and write:
```
ipython lane.py
``` 
![1st_img](https://user-images.githubusercontent.com/65557776/165554145-d8fe5900-fc82-4fae-96fc-78f4d8e3b3a6.png)

- Choose photo from directory and set DEBUGGING_MODE to either true or false
![2nd_img](https://user-images.githubusercontent.com/65557776/165553628-da05b83d-d958-4694-8d05-a3dca29fa3c5.png)

#### Output (if DEBUGGING_MODE = True):
- original image:
![orig](https://user-images.githubusercontent.com/65557776/165554543-19cced7e-3c45-4490-ae0c-f82d6aa2cf72.png)

- Region of interest:
![roi](https://user-images.githubusercontent.com/65557776/165554822-b75304cd-6366-43f1-bccf-71d53a0b2431.png)

- Applied thresholding to region of interest then transformed the image to birdeye view
![transformed_img](https://user-images.githubusercontent.com/65557776/165555561-43cc93ad-2e73-4aac-a642-170549fe7e91.png)

- Warped binary image and histogram of intensity of white pixels in the image
![histogram](https://user-images.githubusercontent.com/65557776/165555878-455158a4-48d9-4b68-b660-80fef9c54e7f.png)

- Using sliding windows technique, we can detect White Pixel in the warped image then we apply polynomial best-fit line through the pixels.
![sliding_win](https://user-images.githubusercontent.com/65557776/165556250-ad803c36-cda6-436b-8804-8dc917145e71.png)

- Filling the lane lines in the search window
![fill_line](https://user-images.githubusercontent.com/65557776/165557771-61b433b5-2bf5-4ab9-ad09-544f8f6dc441.png)

- Overlaying the lane lines with red and blue and the lane with green
![overlayed_lane](https://user-images.githubusercontent.com/65557776/165558253-7b5e269d-b876-4959-92a8-452334255ab3.png)

- last step we calculate center offset and the road curvature radius
![center_pos_and_curve_rad](https://user-images.githubusercontent.com/65557776/165559502-fb740d81-80ab-40c7-a984-e77b3a050fed.png)

- A new directoy should be created called 'output images' and the output image should be saved automatically to this directory
![test1_thresholded](https://user-images.githubusercontent.com/65557776/165561161-26bd3343-ceb7-433a-b00e-f5fad014e6da.jpg)

- Finally we can repeat the process using anyother image in the test_images directory and if you want to quit you can simply type 'q'
![quitting using q](https://user-images.githubusercontent.com/65557776/165561315-73495e12-2318-4ffb-870a-1e5d45483164.png)


### Lane_video notebook (input: Video):
- Download lane_video.py file then open cmd in the directory of the lane.py and write:
```
ipython lane_video.py
```
- Choose video from test_videos directory
![1st_video_input](https://user-images.githubusercontent.com/65557776/165562066-75b4b44e-c9d0-44a6-a519-59785610a479.png)

- The output is a video with overlayed lane lines with red and blue colors and the lane itself with green color
![video_output](https://user-images.githubusercontent.com/65557776/165562747-c2f82b0b-651a-4661-a607-6086b0487be4.png)

- A new directory should be automatically created called output_videos and our output video should be saved there.
![video_output](https://user-images.githubusercontent.com/65557776/165564501-14564228-fed4-4bfe-b33f-07bd1c57a7f2.png)
