import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

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
data_genderandstress = data.iloc[:,[6,11]]
data_genderandstress.rename(columns={'What is your gender?':'Gender','What is your stress level (0-100)?':'Stress'}, inplace=True)
data_genderandstress= data_genderandstress[check_integer(data_genderandstress['Stress'])]
data_genderandstress['Stress'] = pd.to_numeric(data_genderandstress['Stress'])
df_filtered = data_genderandstress[data_genderandstress['Stress'] <=100  ]
df_filtered = df_filtered[df_filtered['Stress'] >=0  ]

df_groupby_gender = df_filtered.groupby('Gender')
df_groupby_gender.hist()
df_groupby_gender.describe()

#select features
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
vY_nb = clf.predict(mX)
print(succes_rate(vY.to_numpy(), vY_nb))

#K-nn
vY=vY.drop(55) #only comes across 1 time which causes errors for KNN
mX=mX.drop(55)
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
#using CV to evaluate score
cv_scores = cross_val_score(knn, mX, vY, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


#using CV to select neighbors
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors' : np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(mX, vY)
#check top performing n_neighbors value
print(knn_gscv.best_params_)
#check mean score for the top performing value of n_neighbors
print(knn_gscv.best_score_)
