
import numpy as np
import cv2

def he(img):
    image = cv2.imread(img, 0)

    histogram, _ = np.histogram(image.ravel(), 256, [0, 256])
    r = image.shape[0] * image.shape[1]
    D = np.zeros((256,))
    for i in range(256):
        for k in range(0, i + 1):
            D[i] += histogram[k]
        D[i] /= r

    n = 0
    while D[n] <= 0:
        n += 1
    min_D = D[n]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = (D[image[i, j]] - min_D) / (1 - min_D)
            image[i, j] = np.floor(255 * temp)
    return image


