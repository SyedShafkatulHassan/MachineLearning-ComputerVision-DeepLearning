

#in this code I have used Flowers Recognition data set from Kagale and predicted flower type using the knn algorithm and open cv . 
#The accuracy rate of the code is 0.667687595712098

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as k
from sklearn.metrics import accuracy_score

im = []
index = []

for i, fo in enumerate((os.listdir('/location'))): #enumerate generate iteam and index and iteam is assainged to fo and index is assainged to i
    fp = os.path.join('/location', fo)
    for fname in os.listdir(fp):
       ip = os.path.join(fp, fname)
       image = cv2.imread(ip)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       image = cv2.resize(image, (100, 100))
       im.append(image)
       index.append(i)
im=np.array(im)
index=np.array(index)
im = im.astype('float32') / 255.0
xt, xte, yt, yte = tts(im, index, test_size=0.38, random_state=35)
knn= k(n_neighbors=3, metric='euclidean')
knn.fit(xt.reshape(len(xt), -1),yt)
yp = knn.predict(xte.reshape(len(xte), -1))
ac = accuracy_score(yte, yp)
print(ac)


