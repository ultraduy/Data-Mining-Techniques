import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv("training_set_VU_DM.csv")
# %%
plt.title('orig_destination_distance')
plt.hist(train['orig_destination_distance'], density=False )

# %%
plt.title('prop_location_score2')
plt.hist(train['prop_location_score2'], density=False )
