from localbinarypatterns import LocalBinaryPatterns
from imutils import paths
import cv2
import pandas as pd
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

db = 'yale'
data_training = 'db/%s' % db

P = 8  # number of points
R = 8  # radius

dataset = [file for file in paths.list_images(data_training)]
desc = LocalBinaryPatterns(8,8)

gbr = paths.list_images(data_training)
nama_gbr = gbr.__next__()
print(nama_gbr)
image = cv2.imread(nama_gbr)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lbp = feature.local_binary_pattern(gray, P, R, method='uniform')
hist = np.histogram(lbp.ravel(), bins=range(0, P+3))
h = hist[0].astype('float')
h /= (h.sum())
print(h, h.sum())
plt.bar(np.array(range(0, P+2)), h)
plt.xticks(np.array(range(0, P+2)))
plt.ylabel('Percentage')
plt.show()