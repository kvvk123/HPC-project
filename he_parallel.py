# parallell implementation on cpu

import cv2
import numpy as np
import multiprocessing
import time
from os import listdir
from os.path import isfile, join


def histogram_equalization(image):
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


if __name__ == '__main__':
    try:
        files = [f for f in listdir("Pict20") if isfile(join("Pict20", f))]

    except FileNotFoundError:
        print('File not found!')
        exit(0)
    t = 0
    for f in files:
        img = cv2.imread(join("Pict20", f), 0)
        start_time = time.time()
        p1 = multiprocessing.Process(target=histogram_equalization, name="P1", args=(1,))
        p2 = multiprocessing.Process(target=histogram_equalization, name="P2", args=(2,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        # converted gray to bgr
        converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # remove the noise from the images
        dst = cv2.fastNlMeansDenoisingColored(converted_img, None, 10, 10, 7, 15)
        # sharpeen the images

        end_time = time.time() - start_time
        t += end_time
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    print("Toal Time Taken =", t)
