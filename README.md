# Simple-Perception-Stack-for-Self-Driving-Cars
In this project we are going to create a simple perception stack for self-driving cars (SDCs.). Our only data source will be video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks.

## How to run the code:

After cloning the repositery and adding test images/videos in their directory 
### run script.sh
- Type either 1 for lane detection or 2 for lane and car detection
![start](https://user-images.githubusercontent.com/65557776/169696603-108f89f5-8893-4c25-a8c1-3c6af0a899a1.png)

### Part 1 (input 1 -> lane detection)
- Select either 1 for image or 2 for video input
![lane photo or vid](https://user-images.githubusercontent.com/65557776/169696701-28b96913-ab71-45b9-9460-d3b1e67c5e83.png)

#### Lane notebook (input 1 -> image): 

- Choose photo from directory and set DEBUGGING_MODE to either true or false
![lane image selection](https://user-images.githubusercontent.com/65557776/169696807-1bf95b44-e142-4299-b48d-46c53a5c49e2.png)

##### Output (if DEBUGGING_MODE = True):
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
![shell_4](https://user-images.githubusercontent.com/65557776/165973206-4ea6a40a-34b4-43af-9896-dbc8d810324d.png)

#### Lane_video notebook (input: Video):
- Run script.sh and select video instead
- Choose video from test_videos directory
![lane video selection](https://user-images.githubusercontent.com/65557776/169697099-82db2535-b85d-4851-8fd0-2f2bed84ebfa.png)

- The output is a video with overlayed lane lines with red and blue colors and the lane itself with green color
![output_saved_video](https://user-images.githubusercontent.com/65557776/165974172-c19fe17d-da5e-4976-a23b-8bf690d2c1aa.png)

- The output video should be saved in  output_videos directory.

Link to output video:
 https://drive.google.com/drive/folders/17ZdWBvtGSpbW7l-AJb6H5U5nit9g6d_m?usp=sharing
 
 
### Part 2 (input 2 -> lane and car detection)
Before running this code you will need to download yolov3.cfg, yolov3.weights, and coco.names then put them in 'part1' directory
- Select either 1 for image or 2 for video input

#### YOLO_object_detection notebook (input 1 -> image):

- Choose image from directory

![lane  and car image](https://user-images.githubusercontent.com/65557776/169697272-93318783-edd4-41c6-a310-b5255f563ad9.png)

#### Output:
- original image:

![orig](https://user-images.githubusercontent.com/65557776/169697419-3e8ce9c2-282c-45a1-bad0-7926ea1adfc6.png)

- Lane detected image

![lane_detected_image](https://user-images.githubusercontent.com/65557776/169697457-e8ea0785-d0b9-4992-9e1f-acb9ebed1edf.png)

- Final image:

![result](https://user-images.githubusercontent.com/65557776/169698087-0067bc39-dd0f-4467-a1f0-00350ffa4084.png)

- Result image is stored in the output_images directory

#### YOLO_object_detection_video notebook (input 2 -> video):
- Choose video from directory

![car detection video](https://user-images.githubusercontent.com/65557776/169698396-966cf5ee-e982-4c3d-b5e7-f4f9219222e6.png)

#### Output:
- Result image is stored in the output_videos directory

![output video](https://user-images.githubusercontent.com/65557776/169698538-2449ce1e-bd30-4d88-9827-83519cbb8c79.png)

Link to output video:
https://drive.google.com/drive/folders/17ZdWBvtGSpbW7l-AJb6H5U5nit9g6d_m?usp=sharing
 



