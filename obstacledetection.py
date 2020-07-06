import numpy as np
import time
import cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
import pyttsx3;
engine =pyttsx3.init();
engine.setProperty('rate', 150);
LABELS = open("coco.names").read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
print("\n\n\n*******************    Welcome To Vision    *******************\n")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture("video1.mp4")
frame_count = 0
start = time.time()
first = True
frames = []

while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = cap.read()
	frames.append(frame)
	if frame_count == 1000:
		break
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 30 == 0:
			end = time.time()
			(H, W) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start= time.time()
			layerOutputs = net.forward(ln)
			end=time.time();
			boxes = []
			confidences = []
			classIDs = []
			centers = []
			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > 0.6:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
			texts = []
			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					#cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			#0.5, color, 2)
					
					centerX, centerY = centers[i][0], centers[i][1]
					if centerX <= W/3:
						W_pos = "left "
					elif centerX <= (W/3 * 2):
						W_pos = "center "
					else:
						W_pos = "right "
					
					if centerY <= H/3:
						H_pos = "top "
					elif centerY <= (H/3 * 2):
						H_pos = "mid "
					else:
						H_pos = "bottom "
					
					texts.append(H_pos + W_pos + LABELS[classIDs[i]])
			#np.rot90(frame)
			cv2.imshow("Image", frame)
			print(texts)
			flag=0
			m=0
			l=0
			r=0
			for i in texts:
				orientation=i.split()
				if orientation[1]=="left":
					l=l+1
				elif orientation[1]=="right":
					r=r+1
				elif orientation[1]=="center":
					m=m+1
			if m!=0:
				if l==0:
					engine.say("move left")
					print("###    move left    ###\n")
					
					engine.runAndWait()
				elif r==0:
					engine.say("move right\n\n")
					print("###    move right   ###\n")
					engine.runAndWait()
				else :
					engine.say("stop\n\n")
					print("###    stop!!    ###\n")
					engine.runAndWait()
			print("Time For this Frame is - ",end-start,"\n\n")



cap.release()
cv2.destroyAllWindows()

