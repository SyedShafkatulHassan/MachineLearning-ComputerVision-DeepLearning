
# Author : Syed Shafkatul Hassan
# Dataset link: https://www.kaggle.com/code/bonnie13000/starter-new-york-city-airbnb-open-data-c675b3c2-a
# Using this dataset I have tried to predict room type using logistic regression
# accuracy_score of the code is 0.791290393631415 


import pandas as pan
import missingno as m
import plotly.express as p
import numpy as n
from sklearn.model_selection import  train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score as ac

da=pan.read_csv("/content/AB_NYC_2019.csv")
da.head()

da  = da.drop('id',axis=1)
da  = da.drop('name',axis=1)
da  = da.drop('host_id',axis=1)
da  = da.drop('host_name',axis=1)
da  = da.drop('last_review',axis=1)
da  = da.drop('neighbourhood_group',axis=1)
da  = da.drop('latitude',axis=1)
da  = da.drop('longitude',axis=1)
da  = da.drop('number_of_reviews',axis=1)
da  = da.drop('reviews_per_month',axis=1)
da  = da.drop('availability_365',axis=1)
da.head()

da.isnull().sum()

m.bar(da, figsize=(10, 6))

da.head()

x = da[['neighbourhood', 'price', 'minimum_nights','calculated_host_listings_count']]
y = da ["room_type"]

xt, xte, yt, yte = tts(x, y, test_size=0.35, random_state=45)

da.head()
da.isnull().sum()

L=LR()
x_train_encoded = pan.get_dummies(xt)
L.fit(x_train_encoded,yt)
PY = L.predict(x_train_encoded)

da.head()

ac = L.score(x_train_encoded, yt)
print("ac:", ac)