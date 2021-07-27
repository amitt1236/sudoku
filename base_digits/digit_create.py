"""
creates (28x28) images of digits, using augmentation to create many
different representation of digits.
The augmentation includes different fonts, rotation and Morphological Transformations
such as erode, opening, closing,and sharpening.
"""

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import imutils
import numpy as np
import os

# A floor value for pixel value sum in each image.
# screening images for garbage.
floor = 30000

dir = "/base_digits/synthetic_digits/"
for i in range(0, 10):
    try:
        os.mkdir(dir + str(i))
    except:
        pass

# rotation anf font augmentation
angle = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
imagesStr = []
fonts = {"Times New Roman.ttf": -3, "Arial.ttf": -3, "Verdana.ttf": -5, "Helvetica.ttc": 0, "Times.ttc": 1
    , "Georgia.ttf": -5, "Palatino.ttc": 0, "Tahoma.ttf": -5, "Trebuchet MS.ttf": -4, "Arial Black.ttf": -8,
         "Comic Sans MS.ttf": -8}

for name, height in fonts.items():

    for i in range(0, 10):
        num = str(i)
        font = ImageFont.truetype(name, 30)
        img = Image.new("RGBA", (28, 28), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((5, height), num, (255, 255, 255), font=font)
        imagesStr.append(num + name)
        img.save(dir + num + "/" + name + ".png")

temp = []
for j in range(len(angle)):
    for i in imagesStr:
        b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
        b = imutils.rotate(b, angle[j])

        temp.append(i + "(" + str(angle[j]) + ")")
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + ".png", b)
imagesStr.extend(temp)


# Morphological Transformations

# 1 erode
for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((2, 2), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_ERODE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "1.1" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((1, 1), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_ERODE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "1.2" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((1, 1), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_ERODE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "1.3" ".png", b)

# 2 dilate
for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((2, 2), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_DILATE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "2.1" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((2, 1), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_DILATE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "2.1" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((1, 2), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_DILATE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "2.3" ".png", b)

# 5 close
for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.ones((3, 3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "5" ".png", b)

# 6 sharpening

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    b = cv2.filter2D(b, -1, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, 0]])
    b = cv2.filter2D(b, -1, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.1" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel2 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    b = cv2.filter2D(b, -1, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.2" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.array([[0, -1, 0],
                       [-1, 7, -1],
                       [-1, -1, -1]])
    b = cv2.filter2D(b, -1, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.3" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    kernel = np.array([[-1, -1, 0],
                       [-1, 9, -1],
                       [0, -1, -1]])
    b = cv2.filter2D(b, -1, kernel)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.4" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    b = cv2.addWeighted(b, 4, cv2.blur(b, (100, 100)), -4, 128)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.5" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    b = cv2.addWeighted(b, 7, cv2.blur(b, (100, 100)), -7, 128)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.6" ".png", b)

for i in imagesStr:
    b = cv2.imread(dir + i[0:1] + "/" + i[1:] + ".png")
    b = cv2.addWeighted(b, 15, cv2.blur(b, (100, 100)), -15, 128)
    if (np.sum(b) > floor):
        cv2.imwrite(dir + i[0:1] + "/" + i[1:] + "(" + str(angle[j]) + ")" + "6.7" ".png", b)
