import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np 
import math  
import time 


cap = cv2.VideoCapture(0) # to access the webcam
detector = HandDetector(maxHands=1) # shows no.of hands
classifier = Classifier("model/keras_model.h5" , "model/labels.txt")

offset = 20 #margin around bbox 
imgsize = 300# size of the white canvas 
folder = "data\A" # where should the images be saved 
labels = ["A"  , "B" , "C" , "D" , "E" , "F" , "G" , "H" , "I" , "J" , "K" , "L" , "M" , "N" , "0" , "P"]
counter = 0 




while True: 
    success, img = cap.read()
    imgOutput = img.copy()
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
                wCal = min(imgsize ,math.ceil(k * w))
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                wGap = (imgsize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
                prediction , index =classifier.getPrediction(imgWhite , draw = False)
            

                print(prediction , index)


            else:
                k = imgsize / w
                hCal = min(imgsize ,math.ceil(k * h))
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                hGap = (imgsize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize
                prediction , index =classifier.getPrediction(imgWhite , draw = False)
            cv2.rectangle(imgOutput , (x-offset,y-offset-50) , (x-offset+150 , y-offset-50+50) , (255 , 255 , 255) , cv2.FILLED)   
            cv2.putText(imgOutput , labels[index] , (x , y-20) ,cv2.FONT_HERSHEY_COMPLEX , 2 ,(255 , 0 , 255) , 2 )
            cv2.rectangle(imgOutput , (x-offset,y-offset) , (x+w+offset, y+h+offset) , (255 , 0 , 255) , 4 )

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
           
           

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
    


