from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 

path = os.path.join(os.getcwd(), 'data', 'faces', 'article', '9.jpg')


image = cv2.imread(path)
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(RGB, (256, 256))
gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rotate = cv2.rotate(image, cv2.ROTATE_180)

imagelist = [image, resized, gry]

fig = plt.figure(figsize=(8,8))
rows = 2
cols = 2

for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    if i == 1:
        plt.title("BGR format")
        plt.imshow(image)
    elif i == 2:
        plt.title("Grayscale")
        plt.imshow(gry, cmap='gray')
    elif i == 3:
        plt.title("Resized BGR")
        plt.imshow(resized)
    elif i == 4:
        plt.title("Rotated 180")
        plt.imshow(rotate)
plt.show()