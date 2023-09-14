
# Author : Syed Shafkatul Hassan
# Dataset link: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
# Using this dataset I have tried to predict if a person has housing or not using the SVM algorithm
# accuracy_score of the code is 0.7173940867784462


import pandas as pan
import missingno as m
import plotly.express as p
import numpy as n
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import  train_test_split as tts
from sklearn.metrics import accuracy_score as ac
from sklearn.svm import SVC as sv
import seaborn as se
from sklearn.preprocessing import StandardScaler as scal

da=pan.read_csv("/content/bank.csv")
da.head()

da  = da.drop('marital',axis=1)
da  = da.drop('contact',axis=1)
da  = da.drop('day',axis=1)
da  = da.drop('month',axis=1)
da  = da.drop('duration',axis=1)
da  = da.drop('campaign',axis=1)
da  = da.drop('pdays',axis=1)
da  = da.drop('previous',axis=1)
da  = da.drop('poutcome',axis=1)
da.head()

da.isnull().sum()

m.bar(da, figsize=(10, 6))

x = da[['age', 'job','education','default','loan','deposit','balance']]
y = da ["housing"]

xt, xte, yt, yte = tts(x, y, test_size=0.3, random_state=45)
xe = pan.get_dummies(xt)
s =scal()
newx = s.fit_transform(xe)

sm = sv(C=10000)

sm.fit(newx, yt)

yp = sm.predict(newx)

ac =  ac(yt, yp)
print("ac:", ac)