#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries 

# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# #### 1- Binary Array 
# return mask in which pixels are either 0 or 1 <br/>
# 0 for black pixels 1 for white pixels

# In[8]:


def binary_array(array, thresh, value=0):
  
  # Value == 0 -> create Array of 1s
  if value == 0:
    binary = np.ones_like(array) 
    
  # Value == 1 -> create Array of 0s  
  else:
    binary = np.zeros_like(array)  
    value = 1

  binary[(array >= thresh[0]) & (array <= thresh[1])] = value 
  return binary


# #### 2-  Reduce noise and details in the image using blur gaussian mask

# In[9]:


def blur_gaussian(channel, ksize=3):
  return cv2.GaussianBlur(channel, (ksize, ksize), 0)


# #### 3- Sobel edge detection
# Detect edges both vertically and horizontally then <br/>
# then we get the result value of from both victors and <br/> pass it to binary_array ()

# In[10]:


# 
# Return Binary (black and white) 2D mask image
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
  
  # Get the magnitude of the edges that are vertically aligned on the image
  sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)
  sobelx = np.absolute(sobelx)
         
  # Get the magnitude of the edges that are horizontally aligned on the image
  sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, sobel_kernel)
  sobely = np.absolute(sobely)
 
  # Find areas of the image that have the strongest pixel intensity changes
  # in both the x and y directions.
  mag = np.sqrt(sobelx ** 2 + sobely ** 2)
 
  # Return a 2D array that contains 0s and 1s   
  return binary_array(mag, thresh)


# #### 4- Apply a threshold to the input channel

# In[11]:


def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
  # If pixel intensity is greater than thresh[0], make that value
  # white (255), else set it to black (0)
  return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


# In[ ]:




