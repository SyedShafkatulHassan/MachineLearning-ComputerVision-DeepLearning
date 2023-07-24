

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as k
from sklearn.metrics import accuracy_score
from google.colab import drive
from sklearn.svm import LinearSVC as LC

im = []
index = []

for i, fo in enumerate((os.listdir(''))):
    fp = os.path.join('', fo)
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
lc = LC()
lc.fit(xt.reshape(len(xt), -1),yt)
yp = lc.predict(xte.reshape(len(xte), -1))
ac = accuracy_score(yte, yp)
print(ac)