import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Assignment 1 DMT
"""

#data exploration
data = pd.read_csv("ODI-2022.csv", delimiter= ";")
print(data.describe())
