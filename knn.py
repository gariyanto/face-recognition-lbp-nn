# import the necessary packages
import itertools
from sklearn.model_selection import train_test_split
from localbinarypatterns import LocalBinaryPatterns
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import cv2
import pandas as pd
import numpy as np


def tulis_hasil(hasil, fname):
    hasil = np.array(hasil)
    pd.DataFrame({"k": hasil[:, 0], "points": hasil[:, 1], "radius": hasil[:, 2],
                  "akurasi": hasil[:, 3]}).to_csv(fname, index=False, header=True)

db = 'jaffe'
data_training = 'db/%s' % db

points = range(8, 70, 4)
radius = range(4, 21)
par = list(itertools.product(points, radius))
hasil = []

dataset = [file for file in paths.list_images(data_training)]




for p in par:
    print("##### points: %d, radius:%d #####" % p)
    # initialize the local binary patterns descriptor
    desc = LocalBinaryPatterns(p[0], p[1])
    data = []
    labels = []

    # loop over the training images
    print(data_training, len(dataset))
    for imagePath in paths.list_images(data_training):
        # load the image, convert it to grayscale, and describe it
        # print("test:", imagePath)
        image = cv2.imread(imagePath)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print('ERROR di:', imagePath)
        hist = desc.describe(gray)
        # extract the label from the image path, then update the
        # label and data lists
        # print(imagePath.split("/"))  # use "\\" in Windows
        labels.append(imagePath.split("/")[-2])  # use "\\" in Windows
        data.append(hist)
    #print(labels)
    print('hist:', len(hist))
    print(hist)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=0)

    # train a Linear KNN on the data
    k_nn = range(1, 10, 2)
    for k in k_nn:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)

        benar = 0
        jml = 0
        for i in range(len(y_test)):
            jml += 1
            hist = X_test[i]
            prediction = neigh.predict([hist])[0]
            if prediction == y_test[i]:
                benar += 1
        akurasi = float(benar*100/jml)
        print(benar, jml, k, ": Akurasi", akurasi, "%")
        hasil.append([k, p[0], p[1], akurasi])
tulis_hasil(hasil, "results/{0}_knn.csv".format(db))