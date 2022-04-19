# %%
import numpy as np
import pandas as pd

# %%
data = pd.read_csv("ODI-2022.csv", delimiter= ";")


# %%
data.describe()        


# %%
data['What programme are you in?'].hist()

# %%
data['Have you taken a course on machine learning?'].hist()

# %%
data['What is your gender?'].hist()

# %%
data['Have you taken a course on statistics?'].hist()

# %%
data['What is your stress level (0-100)?'].hist()

# %%
data_chocoandstress = data.iloc[:,[7,11]]
data_chocoandstress.rename(columns={'Chocolate makes you.....':'Chocolate','What is your stress level (0-100)?':'Stress'}, inplace=True)
data_chocoandstress

# %%
data_chocoandstress['Stress']

# %%
data_chocoandstress.head(85)

# %%
data_chocoandstress = data_chocoandstress.drop(labels=[15,85,95,120,136,277,289], axis=0)

# %%
data_chocoandstress.head(287)

# %%
data_chocoandstress['Stress'] = pd.to_numeric(data_chocoandstress['Stress'])

# %%
df_filtered = data_chocoandstress[data_chocoandstress['Stress'] <=100  ]
df_filtered1 = df_filtered[df_filtered['Stress'] >=0  ]
df_filtered1


# %%
df_groupby_sex = df_filtered1.groupby('Chocolate')
df_groupby_sex.hist()

# %%
df_groupby_sex.describe()

# %%



