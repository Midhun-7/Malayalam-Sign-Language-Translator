import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import ImageFont, ImageDraw,Image
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


font_path = 'Meera.ttf'
font_size = 80
font = ImageFont.truetype(font_path, font_size)

image = np.zeros((100, 300, 3), dtype=np.uint8)
image.fill(255) 


pil_image = Image.fromarray(image)

result_image = np.array(pil_image)



draw = ImageDraw.Draw(pil_image)
# ////
offset = 20
imgSize = 300


counter = 0
num=-1


labels =["അ","ആ","ഇ","ഉ","ഋ"]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # cv2.rectangle(imgOutput, (x - offset, y - offset-50),
        #               (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        # cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        print(labels[index])
        if(num!=index):
            num=index
            draw.rectangle((10,10,200,90),fill=(255,255,255))
            # result_image = np.array(pil_image)
        draw.text((10, 10), labels[index], font=font, fill=(0, 0, 255))  # Black text
        
        # draw.rectangle((10,10,200,30),fill=(255,255,255))

        cv2.rectangle(imgOutput, (x-offset, y-offset),(x + w+offset, y + h+offset), (255, 0, 255), 4)


        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)
        
    result_image = np.array(pil_image)
    cv2.imshow("Malayalam Text", result_image)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(110)
    # Display the image using OpenCV
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

