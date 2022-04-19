# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("train.csv", delimiter= ",")
data

# %%
# Give some insights on descriptive statistics
from pandas import set_option
set_option('precision',2)
description=data.describe()
print(description)

# %%
set_option('precision',3)
correlation=data.corr(method='pearson')
print(correlation)

# %%
class_count=data.groupby('Survived').size()
print(class_count)

# %%
# Percentage of Survived 
percentage=342/891*100
print(percentage)

# %%
### Give 2 of each relations
# 1. Strong Negative Correlation with Response variable
# 2. Strong Positive Correlation with Response variable
# 3. Weak Correlation (negative or Positive) with Response Variable
# 4. Top 2 Strongly Correlated Features
# 5. What do you expect are the top 3 features that influence the response variable

$Ans1 male and pclass
$Ans2 fare and survived 
$Ans3 Age and survived
$Ans4 travel alone male vs fare survived 
$Ans5 fare , male and pclass



# %%
from matplotlib import pyplot 
%matplotlib inline
import matplotlib as plt
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
data.hist()
pyplot.show()
#Ans Age shows normal distribution , fare shows exponential distribution ,survived shows binary distribution , male and travel alone also show binary distribution, categorical  data and hold no significance 


# %%
import matplotlib as plt
fig_size[0]=10
fig_size[1]=8
plt.rcParams["figure.figsize"] = fig_size
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()
#because they are categorical in nature 

# %%
import matplotlib as plt
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
data[['Age','Fare']].plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

# %%
import seaborn as sns
correlation=data.corr()
#plot correlation matrix
fig=pyplot.figure()
cax=sns.heatmap(correlation,vmin=-1,vmax=1,annot= True)
pyplot.show()

# %%
from pandas.plotting import scatter_matrix
fig_size[0] = 10
fig_size[1] = 10
scatter_matrix(data[['Age','Fare']])
pyplot.show()

# %%
data.hist()

# %%
data.info()

# %%
sns.catplot(x='Survived', data=data, kind='count').set(title='Survived')
survived_perc = round((data['Survived'].sum()) / len(data.index) * 100,2)
print(f'Percentage who survived: {survived_perc}%')
plt.show()

# %%
#Pclass
sns.catplot(x='Pclass', data=data, kind='count').set(title='Pclass')
plt.show()

# %%
# PClass and Survived
sns.catplot(x='Pclass', hue='Survived', data=data, kind='count').set(title='Pclass and Survived')
plt.show()

# %%
# Sex
sns.catplot(x='Sex', data=data, kind='count').set(title='Sex')
plt.show()

# %%
# Sex and Survived
sns.catplot(x='Sex', hue='Survived', data=data, kind='count').set(title='Sex and Survived')
plt.show()

# %%



