# -*- coding: utf-8 -*-
"""
@author: han0305
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

games = pd.read_csv('games.csv')
print(games.columns)
print(games.shape)

plt.hist(games["average_rating"])
plt.show()

print(games[games["average_rating"]==0].iloc[0])
print(games[games["average_rating"]>0].iloc[0])

games = games[games["average_rating"]>0]
games = games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show()

corrmat = games.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat,vmax = .8, square = True)
plt.show()

columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]
target = "average_rating"

from sklearn.cross_validation import train_test_split
train = games.sample(frac=0.8, random_state=1)
test = games.iloc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

regressor = LinearRegression()
regressor.fit(train[columns],train[target])

predictions1=regressor.predict(test[columns])
print(mean_squared_error(predictions1,test[target]))

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
rfr.fit(train[columns],train[target])
predictions2=rfr.predict(test[columns])
print(mean_squared_error(predictions2,test[target]))




