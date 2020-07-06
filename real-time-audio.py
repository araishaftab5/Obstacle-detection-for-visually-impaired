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
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	cv2.imshow('my webcam', frame)
	frames.append(frame)

	if frame_count == 300:
		break
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()
			(H, W) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)
			boxes = []
			confidences = []
			classIDs = []
			centers = []
			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > 0.5:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						img5 = cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)  # draw red bounding box in img
						
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

			texts = []
			if len(idxs) > 0:
				for i in idxs.flatten():
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
					cv2.putText(frame, texts[i], (x, y),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
					cv2.imshow('obstacle detection', img5)

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
					engine.runAndWait()
				elif r==0:
					engine.say("move right")
					engine.runAndWait()
				else :
					engine.say("stop")
					engine.runAndWait()

				# if orientation[1]=="center" and (orientation[0]=="mid" or orientation[0]=="bottom"):
				# 	flag=1
					# engine.say("get away")
					# engine.runAndWait()
			
			# if texts:
			# 	description = ', '.join(texts)
			# 	tts = gTTS(description, lang='en')
			# 	#tts.save('tts.mp3')
			# 	engine.say(texts);
			# 	engine.runAndWait();
				#tts = AudioSegment.from_mp3("tts.mp3")
				#subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])


cap.release()
cv2.destroyAllWindows()
#os.remove("tts.mp3")
