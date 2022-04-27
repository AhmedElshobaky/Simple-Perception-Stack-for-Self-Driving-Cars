#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries 

# In[1]:


import cv2
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt
import jdc
import os


# In[2]:


# Class lane represents the car lane and its attributes 
class Lane:
  def __init__(self, orig_frame):
    # original frame with no masks/filters
    self.orig_frame = orig_frame
 
    # Final output of the frame after lane line detection       
    self.lane_line_markings = None
 
    # Frame after each transformation:: used for debugging
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
 
    # original image size
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
 
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height
     
    # Region of interest corners to which the car should be directed
    self.roi_points = np.float32([
      (575,450), # Top-left corner
      (250, 700), # Bottom-left corner            
      (1150,700), # Bottom-right corner
      (735,450) # Top-right corner
    ])
         
    # The desired region of interest after perspective transformation.
    self.padding = int(0.25 * width) # padding from side of the image in pixels
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner         
      [self.orig_image_size[
        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0]-self.padding, 0] # Top-right corner
    ]) 
         
    # Histogram of lane line detection
    self.histogram = None
         
    # Sliding window parameters
    self.no_of_windows = 10
    self.margin = int((1/12) * width)
    self.minpix = int((1/24) * width)
         
    # Best fit polynomial lines for left line and right line of the lane
    self.left_fit = None
    self.right_fit = None
    self.left_lane_inds = None
    self.right_lane_inds = None
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None
         
    # Pixel parameters for x and y dimensions,
    # this is calibrated based on the camera angle
    self.YM_PER_PIX = 10.0 / 1000
    self.XM_PER_PIX = 3.7 / 781
         
    # Radius of curvature and offset to be calculated later
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None


# In[3]:


get_ipython().run_cell_magic('add_to', 'Lane', "\n#Calculate the position of the car relative to the center\ndef calculate_car_position(self, DEBUGGING_MODE=False):\n\n    # Get position of car\n    car_location = self.orig_frame.shape[1] / 2\n \n    # Find the x coordinate of the lane line bottom\n    height = self.orig_frame.shape[0]\n    bottom_left = self.left_fit[0]*height**2 + self.left_fit[1]*height + self.left_fit[2]\n    bottom_right = self.right_fit[0]*height**2 + self.right_fit[1]*height + self.right_fit[2]\n     \n    # calculate center offset relatively to center lane and car location\n    center_lane = (bottom_right - bottom_left)/2 + bottom_left \n    center_offset = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX\n \n    if DEBUGGING_MODE == True:\n      print(str(center_offset) + 'm')\n             \n    self.center_offset = center_offset\n       \n    return center_offset")


# In[4]:


get_ipython().run_cell_magic('add_to', 'Lane', "\n#Bonus: Calculate the road curvature.\ndef calculate_curvature(self, DEBUGGING_MODE=False):\n    \n    # Set the y-value where we want to calculate the road curvature.\n    # Select the maximum y-value\n    y_eval = np.max(self.ploty)    \n \n    # Fit polynomial curves to the real world environment\n    left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (self.XM_PER_PIX), 2)\n    right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (self.XM_PER_PIX), 2)\n             \n    # Calculate the radius of curvature of the lane of the car\n    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])\n    right_curvem = ((1 + (2*right_fit_cr[0]*y_eval*self.YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])\n     \n    # Display on terminal window\n    if DEBUGGING_MODE == True:\n      print(left_curvem, 'm', right_curvem, 'm')\n             \n    self.left_curvem = left_curvem\n    self.right_curvem = right_curvem\n \n    return left_curvem, right_curvem")


# In[5]:


get_ipython().run_cell_magic('add_to', 'Lane', '# histogram of the image to find peaks of white pixels\ndef calculate_histogram(self,frame=None,plot=True):\n\n    if frame is None:\n      frame = self.warped_frame\n             \n    # Generate the histogram\n    self.histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)\n \n    if plot == True:\n         \n      # Draw both the image and the histogram\n      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns\n      figure.set_size_inches(10, 5)\n      ax1.imshow(frame, cmap=\'gray\')\n      ax1.set_title("Warped Binary Frame")\n      ax2.plot(self.histogram)\n      ax2.set_title("Histogram Peaks")\n      plt.subplots_adjust(hspace=0.4)\n      plt.show()\n             \n    return self.histogram')


# In[6]:


get_ipython().run_cell_magic('add_to', 'Lane', '# put curvature and offset text on the image\ndef display_curvature_offset(self, frame=None, plot=False): \n    image_copy = None\n    if frame is None:\n      image_copy = self.orig_frame.copy()\n    else:\n      image_copy = frame\n \n    cv2.putText(image_copy,\'Curve Radius: \'+str((self.left_curvem+self.right_curvem)/2)[:7]+\' m\', \n      (int((5/600)*self.width), int((20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, \n      (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)\n    \n    if self.center_offset == 0:\n        cv2.putText(image_copy,\'Car is centered\', \n          (int((5/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, \n          (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)\n    elif self.center_offset > 0:\n        cv2.putText(image_copy,\'Car is \'+str(self.center_offset*100)[:7]+\' cm to the right\', \n          (int((5/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, \n          (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)\n            \n    elif self.center_offset < 0:\n        cv2.putText(image_copy,\'Car is \'+str(np.abs(self.center_offset)*100)[:7]+\' cm to the left\', \n          (int((5/600)*self.width), int((40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, \n          (float((0.5/600)*self.width)),(255,255,255),2,cv2.LINE_AA)\n             \n    if plot==True:       \n      cv2.imshow("Image with Curvature and Offset", image_copy)\n      cv2.waitKey(0)\n \n    return image_copy')


# In[7]:


get_ipython().run_cell_magic('add_to', 'Lane', '\n# Obtain parameters to calculate polyfit() of the lane line \ndef get_lane_line_previous_window(self, left_fit, right_fit, plot=False):\n\n    # Sliding window parameter\n    margin = self.margin\n \n    # Find the x and y coordinates of all white pixels in the frame.         \n    nonzero = self.warped_frame.nonzero()  \n    nonzeroy = np.array(nonzero[0])\n    nonzerox = np.array(nonzero[1])\n         \n    # Store left and right lane pixel indices\n    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +left_fit[2] - margin)) & \n                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))\n    \n    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &\n                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))\n    \n    self.left_lane_inds = left_lane_inds\n    self.right_lane_inds = right_lane_inds\n \n    # Get the left lane line pixel locations  \n    leftx = nonzerox[left_lane_inds]\n    lefty = nonzeroy[left_lane_inds]\n    # Get the right lane line pixel locations\n    rightx = nonzerox[right_lane_inds]\n    righty = nonzeroy[right_lane_inds]  \n \n    self.leftx = leftx\n    self.rightx = rightx\n    self.lefty = lefty\n    self.righty = righty        \n     \n    # Fit a second order polynomial curve to each lane line\n    left_fit = np.polyfit(lefty, leftx, 2)\n    right_fit = np.polyfit(righty, rightx, 2)\n    \n    self.left_fit = left_fit\n    self.right_fit = right_fit\n         \n    # Create the x and y values to plot on the image\n    ploty = np.linspace(0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0])\n    \n    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]\n    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]\n    \n    self.ploty = ploty\n    self.left_fitx = left_fitx\n    self.right_fitx = right_fitx\n         \n    if plot==True:\n         \n      # draw images to draw the lane on them\n      out_img = np.dstack((self.warped_frame, self.warped_frame, (self.warped_frame)))*255\n      window_img = np.zeros_like(out_img)\n             \n      # coloring left and right line pixels\n      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]\n      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]\n      \n      # Create a polygon on the search window area and cast right and left points to use in cv2.fillPoly()\n      margin = self.margin\n    \n      #left line\n      left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])\n      left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])\n      left_line_pts = np.hstack((left_line_window1, left_line_window2))\n      \n      #right line\n      right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])\n      right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])\n      right_line_pts = np.hstack((right_line_window1, right_line_window2))\n             \n      # color the lane on the warped image\n      cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))\n      cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))\n      \n      result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)\n       \n      # Plot the figures \n      figure, ( ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column\n      figure.set_size_inches(10, 10)\n      figure.tight_layout(pad=3.0)\n      \n      ax1.imshow(self.warped_frame, cmap=\'gray\')\n      ax2.imshow(result)\n      \n      ax2.plot(left_fitx, ploty, color=\'yellow\')\n      ax2.plot(right_fitx, ploty, color=\'yellow\')\n       \n      ax1.set_title("Warped Frame")\n      ax2.set_title("Warped Frame With Search Window")\n      plt.show()')


# In[8]:


get_ipython().run_cell_magic('add_to', 'Lane', '\n#Locate lane line pixels indices\ndef get_lane_line_indices_sliding_windows(self, plot=False):\n    \n    # Sliding window parameter\n    margin = self.margin\n \n    frame_sliding_window = self.warped_frame.copy()\n \n    window_height = np.int(self.warped_frame.shape[0]/self.no_of_windows)       \n  \n    nonzero = self.warped_frame.nonzero()\n    nonzeroy = np.array(nonzero[0])\n    nonzerox = np.array(nonzero[1]) \n         \n    # Array to store indices of the lane  \n    left_lane_inds = []\n    right_lane_inds = []\n         \n    # Current positions for pixel indices for each window to be update laer in the for loop \n    leftx_base, rightx_base = self.histogram_peak()\n    leftx_current = leftx_base\n    rightx_current = rightx_base\n \n    no_of_windows = self.no_of_windows\n         \n    for window in range(no_of_windows):\n       \n      #window boundaries\n      win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height\n      win_y_high = self.warped_frame.shape[0] - window * window_height\n      win_xleft_low = leftx_current - margin\n      win_xleft_high = leftx_current + margin\n      win_xright_low = rightx_current - margin\n      win_xright_high = rightx_current + margin\n      \n      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (255,255,255), 2)\n      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (255,255,255), 2)\n \n      # Identify the white pixels in x and y within the window\n      good_left_inds = ((nonzeroy >= win_y_low) & \n                        (nonzeroy < win_y_high) & \n                        (nonzerox >= win_xleft_low) & \n                        (nonzerox < win_xleft_high)).nonzero()[0]\n        \n      good_right_inds = ((nonzeroy >= win_y_low) &\n                         (nonzeroy < win_y_high) & \n                         (nonzerox >= win_xright_low) & \n                         (nonzerox < win_xright_high)).nonzero()[0]\n                                                         \n      left_lane_inds.append(good_left_inds)\n      right_lane_inds.append(good_right_inds)\n         \n      # If you the indices > minpix pixels, recenter next window on the mean\n      minpix = self.minpix\n      if len(good_left_inds) > minpix:\n        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))\n      if len(good_right_inds) > minpix:        \n        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))\n                     \n    left_lane_inds = np.concatenate(left_lane_inds)\n    right_lane_inds = np.concatenate(right_lane_inds)\n \n    # Get the pixel coordinates for the lane lines\n    leftx = nonzerox[left_lane_inds]\n    lefty = nonzeroy[left_lane_inds] \n    rightx = nonzerox[right_lane_inds] \n    righty = nonzeroy[right_lane_inds]\n \n    # Polyfit() for pixel coordinates of the lane line\n    left_fit = np.polyfit(lefty, leftx, 2)\n    right_fit = np.polyfit(righty, rightx, 2) \n         \n    self.left_fit = left_fit\n    self.right_fit = right_fit\n \n    #Plotting \n    if plot==True:\n         \n      ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])\n    \n      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]\n      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]\n \n      out_img = np.dstack((frame_sliding_window, frame_sliding_window, (frame_sliding_window))) * 255\n             \n      # coloring left and right line pixels\n      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]\n      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]\n                 \n      # Plot the figure with the sliding windows\n      figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column\n      figure.set_size_inches(10, 10)\n      figure.tight_layout(pad=3.0)\n      \n      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))\n      ax2.imshow(frame_sliding_window, cmap=\'gray\')\n      ax3.imshow(out_img)\n    \n      ax3.plot(left_fitx, ploty, color=\'yellow\')\n      ax3.plot(right_fitx, ploty, color=\'yellow\')\n      \n      ax1.set_title("Original Frame")  \n      ax2.set_title("Warped Frame with Sliding Windows")\n      ax3.set_title("Detected Lane Lines with Sliding Windows")\n      plt.show()        \n             \n    return self.left_fit, self.right_fit')


# In[9]:


get_ipython().run_cell_magic('add_to', 'Lane', '\n#applying sobal edge detection to the frame and detect the lane lines\ndef get_line_markings(self, frame=None):\n\n    if frame is None:\n      frame = self.orig_frame\n             \n    # Convert color system from BGR to HLS\n    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)\n\n    # Applying sobel edge detection on the L (lightness) channel of frame\n    # then reducing noise using gaussian filter\n    l_channel = hls[:, :, 1]\n    _, sxbinary = edge.threshold(l_channel, thresh=(120, 255))\n    sxbinary = edge.blur_gaussian(sxbinary, ksize=3)\n    #Applying sobal edge detection \n    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))\n \n    # Applying binary thresholding on the S (saturation) channel of frame\n    s_channel = hls[:, :, 2]\n    _, s_binary = edge.threshold(s_channel, thresh=(80, 255))\n     \n    # Applying binary thresholding on the R (red) channel of frame \n    r_channel = frame[:, :, 2]\n    _, r_thresh = edge.threshold(r_channel, thresh=(120, 255))\n \n    # Bitwise AND to reduce noise    \n    rs_binary = cv2.bitwise_and(s_binary, r_thresh)\n \n    # Combine the all lane lines with the lane line edges using bitwise or\n    self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(\n                              np.uint8))    \n    return self.lane_line_markings')


# In[10]:


get_ipython().run_cell_magic('add_to', 'Lane', '\n#calculate the right and left peak of the histogram\ndef histogram_peak(self):\n    \n    midpoint = np.int(self.histogram.shape[0]/2)\n    leftx_base = np.argmax(self.histogram[:midpoint])\n    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint\n\n    return leftx_base, rightx_base')


# In[11]:


get_ipython().run_cell_magic('add_to', 'Lane', '\n# draw lines of the lane on the frame\ndef overlay_lane_lines(self, plot=False):\n    \n    # Generate an image to draw the lane lines on \n    warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)\n    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       \n         \n    # Recast the x and y points into usable format for cv2.fillPoly()\n    pts_left = np.array([np.transpose(np.vstack([\n                         self.left_fitx, self.ploty]))])\n    pts_right = np.array([np.flipud(np.transpose(np.vstack([\n                          self.right_fitx, self.ploty])))])\n    pts = np.hstack((pts_left, pts_right))\n         \n    # Draw lane on the warped blank image\n    cv2.fillPoly(color_warp,np.int_([pts]), (0,255, 0))\n    cv2.polylines(color_warp, np.int_([pts_left]), True, (255,0,0),10 )\n    cv2.polylines(color_warp, np.int_([pts_right]), True, (0,0,255),10 )\n    \n    # Warp the blank back to original image space using inverse perspective \n    # matrix (Minv)\n    newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (\n                                  self.orig_frame.shape[\n                                  1], self.orig_frame.shape[0]))\n     \n    # Combine the result with the original image\n    result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)\n         \n    if plot==True:\n      \n      # Plot the figures \n      figure, ( ax1) = plt.subplots(1,1) # 1 rows, 1 column\n      figure.set_size_inches(10, 10)\n      figure.tight_layout(pad=3.0)\n    \n      ax1.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))\n      ax1.set_title("Original Frame With Lane Overlay")\n      plt.show()   \n \n    return result ')


# In[12]:


get_ipython().run_cell_magic('add_to', 'Lane', "\n#transform the perspective of the binary image to birds eye view\ndef perspective_transform(self, frame=None, plot=False):\n\n    if frame is None:\n      frame = self.lane_line_markings\n             \n    # Calculate the transformation matrix\n    self.transformation_matrix = cv2.getPerspectiveTransform(\n      self.roi_points, self.desired_roi_points)\n \n    # Calculate the inverse transformation matrix           \n    self.inv_transformation_matrix = cv2.getPerspectiveTransform(\n      self.desired_roi_points, self.roi_points)\n \n    # Perform the transform using the transformation matrix\n    self.warped_frame = cv2.warpPerspective(\n      frame, self.transformation_matrix, self.orig_image_size, flags=(\n     cv2.INTER_LINEAR)) \n \n    # Convert image to binary\n    (thresh, binary_warped) = cv2.threshold(\n      self.warped_frame, 127, 255, cv2.THRESH_BINARY)           \n    self.warped_frame = binary_warped\n \n    # Display the perspective transformed (i.e. warped) frame\n    if plot == True:\n      warped_copy = self.warped_frame.copy()\n      warped_plot = cv2.polylines(warped_copy, np.int32([\n                    self.desired_roi_points]), True, (147,20,255), 3)\n \n      # Display the image\n      while(1):\n        cv2.imshow('Warped Image', warped_plot)\n             \n        # Press any key to stop\n        if cv2.waitKey(0):\n          break\n \n      cv2.destroyAllWindows()   \n             \n    return self.warped_frame ")


# In[13]:


get_ipython().run_cell_magic('add_to', 'Lane', "\n#plotting our region of interest \ndef plot_roi(self, frame=None, plot=False):\n\n    if plot == False:\n      return\n             \n    if frame is None:\n      frame = self.orig_frame.copy()\n \n    # Overlay trapezoid on the frame\n    this_image = cv2.polylines(frame, np.int32([self.roi_points]), True, (255,0,255), 3)\n \n    #display the image\n    cv2.imshow('ROI Image', this_image)\n    cv2.waitKey(0)")


# In[14]:


while 1:
    files = os.listdir('test_images')
    print("======================================")
    print("=         TEST Images           =")
    print("======================================")
    for i in files:
        print('{}\t '.format(i), end='')
        if files.index(i) % 3 == 0 and files.index(i) != 0:
            print('\n')
    print("\n======================================")

    # need to select image name with the extension (ex: img1.jpeg)1
    file = input("Select a file from the directory(q- quit): ").strip()
    # quit program
    if file == 'q' or file == 'Q':
        break
    image = 'test_images/' + file
    # Debug mode
    DEBUGGING_MODE = bool(input("DEBUGGING_MODE: -True -False ").strip())
    
    
    original_frame = cv2.imread(image)
    cv2.imshow('image', original_frame)
    cv2.waitKey(0)
    # Create a Lane object
    lane_obj = Lane(orig_frame=original_frame)
    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()
 
    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=DEBUGGING_MODE)
    # Perform the perspective transform to generate a bird's eye view
    # If Plot == True, show image with new region of interest
    warped_frame = lane_obj.perspective_transform(plot=DEBUGGING_MODE)
    print()
    print()
    # Generate the image histogram to serve as a starting point
    # for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=DEBUGGING_MODE)
    # Find lane line pixels using the sliding window method 
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=DEBUGGING_MODE)
    # Fill in the lane line
    lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=DEBUGGING_MODE)
    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=DEBUGGING_MODE)
    
    # Calculate lane line curvature (left and right lane lines)
    lane_obj.calculate_curvature(DEBUGGING_MODE)
    
    # Calculate center offset                                                                 
    lane_obj.calculate_car_position(DEBUGGING_MODE)
    # Display curvature and center offset on image
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=True)
     
     
    # save image containing highlighted defect
    cv2.imwrite('output_images/{}_thresholded.jpg'.format(file.split('.')[0]),frame_with_lane_lines2)     
    # Close all windows
    cv2.destroyAllWindows()


# In[ ]:




