
Author : Syed Shafkatul Hassan

import pandas as pan
import missingno as m
import plotly.express as p
import numpy as n
from sklearn.cluster import KMeans as KMN
from sklearn.preprocessing import StandardScaler as scal
import matplotlib.pyplot as p
import seaborn as s
from sklearn.metrics import accuracy_score as ac

da=pan.read_csv("/content/Iris.csv")
da.head()

da  = da.drop('Id',axis=1)

da.isnull().sum()

m.bar(da, figsize=(10, 6))

da.head()

s.swarmplot(data=da, x='Species', y='SepalLengthCm')

x = da[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = da['Species']

s =scal()
newx = s.fit_transform(x)

kmn = KMN(n_clusters=3, random_state=30)

kmn.fit(newx)

l = kmn.labels_

cc = kmn.cluster_centers_

p.scatter(newx[:, 0], newx[:, 1], c=l, cmap='viridis')
p.scatter(cc[:, 0], cc[:, 1], marker='x', color='green', label='Cluster Cententroid')
p.title('Iris Dataset K-means Clustering')
p.legend()
p.show()

lm = {0: pan.Series.mode(y[l == 0])[0],
                 1: pan.Series.mode(y[l == 1])[0],
                 2: pan.Series.mode(y[l == 2])[0]}
predict_y = pan.Series(l).map(lm)

accu = ac(y,predict_y)
print("accu:", accu)