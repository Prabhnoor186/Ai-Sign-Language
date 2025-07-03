
import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math  
import time 
import os 

cap = cv2.VideoCapture(0) # to access the webcam
detector = HandDetector(maxHands=1) # shows no.of hands

offset = 20 #margin around bbox 
imgsize = 300# size of the white canvas 
folder = "data/M" # whre should be the images saved s
counter = 0 

while True: 
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands: 
        hand = hands[0] # only using one hand 
        x, y, w, h = hand['bbox'] # gwt bounding box

        # Boundary correction crop stays in the image
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2] # crop region arounnd hand 

        if imgCrop.size != 0: 
            imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgsize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                wGap = (imgsize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgsize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                hGap = (imgsize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
           
            os.makedirs(folder, exist_ok=True)

    cv2.imshow("Image", img)
    key= cv2.waitKey(1)
    if key == ord("s"):
        counter += 1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg'  , imgWhite)
        print(counter) # how many images we saved 



