#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import imutils
import time
import cv2
from imutils.video import VideoStream
from imutils.video import FPS


# In[2]:


path = 'yolo-files'
LabelsPath = os.path.join(path,'coco.names') 


# In[3]:


labels = open(LabelsPath).read().strip().split('\n')


# In[6]:


np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(labels),3), dtype = 'uint8')


# In[7]:


WeightsPath = os.path.join(path,'yolov3.weights')
ConfigPath = os.path.join(path,'yolov3.cfg')


# In[17]:


net = cv2.dnn.readNetFromDarknet(ConfigPath,WeightsPath)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
while True:
    frame = vs.read()
    (H,W) = frame.shape[:2]
    frame = imutils.resize(frame, width = 400)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layeroutput = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layeroutput:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5:
                
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype('int')
                
                x = int(centerX - (width/2))
                y = int(centerY - (width)/2)
                
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
    
    if len(idxs)>0:
        
        for i in idxs.flatten():
            
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            
            text = '{}:{:.4f}'.format(labels[classIDs[i]],confidences[i])
            
            cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    fps.update()
    
fps.stop()

cv2.destroyAllWindows()
vs.stop()
            
                
            
                


# In[ ]:




