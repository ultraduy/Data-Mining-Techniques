import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_integer(data):
    checks = [False] * len(data.index)
    data = data.to_numpy()
    for i in range(len(data)):
        if data[i].isdigit():
            checks[i] = True
    return(checks)
"""
Assignment 1 DMT
"""

#data exploration
data = pd.read_csv("ODI-2022.csv", delimiter= ";")
print(data.describe())

pre_data = data[["What is your gender?","What is your stress level (0-100)?", "Give a random number"]]

#select integers (preprocessing)
pre_data = pre_data[check_integer(pre_data["Give a random number"])]
pre_data = pre_data[check_integer(pre_data["What is your stress level (0-100)?"])]

pre_data["What is your stress level (0-100)?"] = pre_data["What is your stress level (0-100)?"].astype('float')
pre_data["Give a random number"] = pre_data["Give a random number"].astype('float')

pre_data = pre_data[pre_data["What is your stress level (0-100)?"]<=100]
pre_data = pre_data[pre_data["What is your stress level (0-100)?"]>=0]


#obtain final data
vY = pre_data["What is your gender?"]

mX = pre_data[["What is your stress level (0-100)?", "Give a random number"]]
mX = mX.astype('int')

