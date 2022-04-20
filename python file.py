import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
def check_integer(data):
    checks = [False] * len(data.index)
    data = data.to_numpy()
    for i in range(len(data)):
        try:
            float(str(data[i]))
            checks[i] = True
        except:
            checks[i] = False
    return(checks)
def succes_rate(vY, vY_hat):
    count = 0
    for i in range(len(vY)):
        if (str(vY[i]) == str(vY_hat[i])):
            count = count + 1
    return count/len(vY)
"""
Assignment 1 DMT
"""

#data exploration
data = pd.read_csv("ODI-2022.csv", delimiter= ";")
print(data.describe())

pre_data = data[["What is your gender?","What is your stress level (0-100)?", "Give a random number", "You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "]]

#select integers (preprocessing)
pre_data = pre_data[check_integer(pre_data["Give a random number"])]
pre_data = pre_data[check_integer(pre_data["What is your stress level (0-100)?"])]
pre_data = pre_data[check_integer(pre_data[ "You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "])]


pre_data["What is your stress level (0-100)?"] = pre_data["What is your stress level (0-100)?"].astype('float')
pre_data["Give a random number"] = pre_data["Give a random number"].astype('float')
pre_data["You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "] = pre_data["You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "].astype('float')


pre_data = pre_data[pre_data["What is your stress level (0-100)?"]<=100]
pre_data = pre_data[pre_data["What is your stress level (0-100)?"]>=0]

pre_data = pre_data[pre_data["You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "]<=100]
pre_data = pre_data[pre_data["You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "]>=0]

#for naive bayes we can only use numbers larger than 0
pre_data = pre_data[pre_data["Give a random number"].between(-100000000, 100000000)]

#obtain final data
vY = pre_data["What is your gender?"]

mX = pre_data[["What is your stress level (0-100)?", "Give a random number", "You can get 100 euros if you win a local DM competition, or we don’t hold any competitions and I give everyone some money (not the same amount!). How much do you think you would deserve then? "]]
mX = mX.astype('int')

#Naive Bayes
clf = MultinomialNB()
clf.fit(mX, vY)
y_hat = clf.predict(mX)
print(succes_rate(vY.to_numpy(), y_hat))