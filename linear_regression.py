
Author : Syed Shafkatul Hassan



import pandas as pan
import missingno as m
import plotly.express as p
import numpy as n
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import  train_test_split as tts

da=pan.read_csv("/content/winequality-red.csv")
da.head()

da.shape

da.isnull().sum()

m.bar(da, figsize=(10, 6))

p.violin(da, x="pH", y="alcohol", width=1500, height=800)

xs = da[['volatile acidity','citric acid',"citric acid","residual sugar","chlorides",	"free sulfur dioxide","total sulfur dioxide",	"density",	"pH",	"sulphates",	"alcohol",	"quality"]]
ys = da['fixed acidity']

xt, xte, yt, yte = tts(xs, ys, test_size=0.3, random_state=46)

m=LR()
m.fit(xt,yt)
predicty = m.predict(xt)
m = mse(yt, predicty)
rmse = n.sqrt(m)
print('rmse', rmse)