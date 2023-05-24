#https://www.youtube.com/watch?v=1biHdzEkAJg&list=PL0lO_mIqDDFUAQ2RdAgLp6Tj_fREcxk6T&index=7
import cv2
import numpy as np

img = cv2.imread('images/put.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces.xml')

res = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)

for (x, y, w, h) in res:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)


cv2.imshow('test', img)
cv2.waitKey(0)