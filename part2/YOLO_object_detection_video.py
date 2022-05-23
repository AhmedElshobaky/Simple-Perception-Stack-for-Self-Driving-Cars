#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
import edge_detection as edge
import lane_modified


# #### Loading Yolo weights, and config

# In[2]:


weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
names = net.getLayerNames()


# In[3]:


layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]


# In[4]:


labels_path = 'coco.names'
labels = open(labels_path).read().strip().split("\n")


# In[5]:


def videoHandler(vid_dir, vid_res = (1280,720), Debug = False):
    # variables to handle video frames
    
    prev_leftx = None
    prev_lefty = None
    prev_rightx = None
    prev_righty = None   
    prev_left_fit = []
    prev_right_fit = []

    prev_leftx2 = None
    prev_lefty2 = None
    prev_rightx2 = None
    prev_righty2 = None
    prev_left_fit2 = []
    prev_right_fit2 = []

    output_vid_dir = '../output_videos/{}_part2_thresholded.mp4'.format(vid_dir[15:].split('.')[0])
    output_frames_per_second = 20.0                                                       
 
    # Load a video
    capture = cv2.VideoCapture(vid_dir)

    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_vid_dir,fourcc,output_frames_per_second,vid_res)
        
    while capture.isOpened():
        # Capture one frame at a time
        success, frame = capture.read() 
        # Do we have a video frame? If true, proceed.
        if success:
            # Resize the frame
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            frame = cv2.resize(frame, (width, height))
            # Store the original frame
            original_frame = frame.copy()
            lane_detecticted_img = lane_modified.lane_detection(original_frame)
            lane_detecticted_img = cv2.cvtColor(lane_detecticted_img, cv2.COLOR_BGR2RGB)
            (H,W) = lane_detecticted_img.shape[:2]
          
            blob = cv2.dnn.blobFromImage(lane_detecticted_img, 1/255.0, (416,416), crop=False, swapRB = False)
            net.setInput(blob)
            # calculate the runtime of the algorithm
            start_t = time.time()
            layers_output = net.forward(layers_names)
            print("A forward pass through yolov3 took {}".format(time.time() - start_t))
          
            boxes = []
            confidences = []
            classIDs = []
            
            for output in layers_output:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    if(confidence > 0.85):
                        box = detection[:4] * np.array([W,H,W,H])
                        bx,by,bw,bh = box.astype("int")

                        x = int(bx - (bw/2))
                        y = int(by - (bh/2))

                    
                        boxes.append([x,y,int(bw),int(bh)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                   
            
            idxs = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=0.4,nms_threshold=0.6)
            
            labels_path = 'coco.names'
            labels = open(labels_path).read().strip().split("\n")
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x,y) = [boxes[i][0],boxes[i][1]]
                    (w,h) = [boxes[i][2],boxes[i][3]]
    
                    cv2.rectangle(lane_detecticted_img,(x,y),(x+w,y+h),(255,165,0),2)
                    cv2.putText(lane_detecticted_img,"{}: {}".format(labels[classIDs[i]],confidences[i]), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,165,0),2)
            
            lane_detecticted_img = cv2.cvtColor(lane_detecticted_img, cv2.COLOR_BGR2RGB)
            result.write(lane_detecticted_img)
            cv2.imshow("Frame", lane_detecticted_img)
          
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # No more video frames left
        else:
            break
    # Stop when the video is finished
    capture.release()
    # Release the video recording
    result.release()
    # Close all windows
    cv2.destroyAllWindows()


# In[7]:


while 1:
    files = os.listdir('../test_videos')
    print("======================================")
    print("=         TEST Videos           =")
    print("======================================")
    for i in files:
        print('{}\t '.format(i), end='')
        if files.index(i) % 3 == 0 and files.index(i) != 0:
            print('\n')
    print("\n======================================")

    # need to select video name with the extension (ex: project_video.mp4)
    file = input("Select a Video from the directory(q- quit): ").strip()
    # quit program
    if file == 'q' or file == 'Q':
        break
    vid = '../test_videos/' + file
    videoHandler(vid)


# In[ ]:




