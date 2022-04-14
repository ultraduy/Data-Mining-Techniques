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

vY = data["What is your gender?"]
mX = data[["What is your stress level (0-100)?", "Give a random number"]]

#mX = mX[pd.to_numeric(mX["Give a random number"], errors='coerce').notnull()]
#mX = mX[pd.to_numeric(mX["What is your stress level (0-100)?"], errors='coerce').notnull()]

#select integers
mX = mX[check_integer(mX["Give a random number"])]
mX = mX[check_integer(mX["What is your stress level (0-100)?"])]

mX = mX.astype('float')
mX = mX[mX["What is your stress level (0-100)?"]<=100]
mX = mX[mX["What is your stress level (0-100)?"]>=0]

mX = mX.astype('int')