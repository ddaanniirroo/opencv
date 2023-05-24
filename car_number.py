#https://www.youtube.com/watch?v=HA1IcFYU-8Q&list=PL0lO_mIqDDFUAQ2RdAgLp6Tj_fREcxk6T&index=8
import cv2
import numpy as np
import pytesseract

import imutils
from matplotlib import pyplot as pl
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

img = cv2.imread("images/merc.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(gray, 5, 9, 9)
edges = cv2.Canny(img_filter, 30, 40)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        pos = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

x, y = np.where(mask==255)
x1, y1 = np.min(x), np.min(y)+10
x2, y2 = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]

text = pytesseract.image_to_string(crop)
if text:
    print(text)
else:
    print(1)

pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
pl.show()
