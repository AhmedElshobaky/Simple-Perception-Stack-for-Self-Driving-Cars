# Simple-Perception-Stack-for-Self-Driving-Cars
In this project we are going to create a simple perception stack for self-driving cars (SDCs.). Our only data source will be video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks.

## How to run the code:

After cloning the repositery and adding test images/videos in their directory 
### run script.sh
- Type either image or video based on the input

![shell_1](https://user-images.githubusercontent.com/65557776/165972137-eb18714a-1126-4e15-a735-1512d4192e94.png)

### Lane notebook (input: image): 
![shell_2](https://user-images.githubusercontent.com/65557776/165972508-a7e7b474-b1b5-4b7f-91ce-58b74750466b.png)

- Choose photo from directory and set DEBUGGING_MODE to either true or false
![shell_3](https://user-images.githubusercontent.com/65557776/165972780-05e4fadf-d7c1-4e09-928e-cac6859c2fd5.png)

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
![shell_4](https://user-images.githubusercontent.com/65557776/165973206-4ea6a40a-34b4-43af-9896-dbc8d810324d.png)

### Lane_video notebook (input: Video):
- Run script.sh and write video instead
- Choose video from test_videos directory
![shell_5](https://user-images.githubusercontent.com/65557776/165973940-1924bf86-47d3-4155-9fb0-770da9f55e38.png)

- The output is a video with overlayed lane lines with red and blue colors and the lane itself with green color
![output_saved_video](https://user-images.githubusercontent.com/65557776/165974172-c19fe17d-da5e-4976-a23b-8bf690d2c1aa.png)

- A new directory should be automatically created called output_videos and our output video should be saved there.
![overlayed_lane_video](https://user-images.githubusercontent.com/65557776/165567367-bcd9243b-b462-42f2-8bbf-72cc7c70b7c1.png)

Link to output video:
 https://drive.google.com/drive/folders/17ZdWBvtGSpbW7l-AJb6H5U5nit9g6d_m?usp=sharing
