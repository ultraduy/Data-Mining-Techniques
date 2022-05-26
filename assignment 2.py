# %%
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

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %%
train = pd.read_csv("training_set_VU_DM.csv")
train['prop_review_score'] = train['prop_review_score'].fillna(0)
train['prop_location_score2'] = train['prop_location_score2'].fillna(train['prop_location_score2'].mean())
train['orig_destination_distance'] = train['orig_destination_distance'].fillna(train['orig_destination_distance'].median())
# %%
# count of null values in each column
print(train.isnull().sum())

# %%
train = train.dropna(axis="columns")
print(train.isnull().sum())

train = train.dropna(axis=0)
print(train.isnull().sum())
y_train = train["booking_bool"]
X_train = train.drop(["click_bool","position","date_time","booking_bool"],
                     axis=1,
                     inplace=False)



# %%

def optimal_recommendations(model, test_data):
    result = pd.DataFrame(columns=["srch_id","prop_id","scores"])
    df_list = []
    grouped_data = test_data.groupby("srch_id")
    for group_name, group in tqdm(grouped_data):
        mX = group.loc[:, ~group.columns.isin(["srch_id"])]
        mX = mX.drop("prop_id", axis=1, inplace=False)
        scores = model.predict_proba(mX)
        scores = scores[:,1]
        sorted_group = sort_properties(group_name, scores, group["prop_id"])
        df_list.append(sorted_group)
    result = pd.concat(df_list)
    return result

def byUserReview(test_data):
    result = pd.DataFrame(columns=["srch_id","prop_id","scores"])
    df_list = []
    grouped_data = test_data.groupby("srch_id")
    for group_name, group in tqdm(grouped_data):
        scores = group['prop_review_score']
        sorted_group = sort_properties(group_name, scores.values, group["prop_id"])
        df_list.append(sorted_group)
    result = pd.concat(df_list)
    return result

def sort_properties(srch_id, scores, prop_id):
    sorted_group = pd.DataFrame(columns=["srch_id","prop_id","scores"])
    sorted_group["srch_id"] = len(scores)*[srch_id]
    sorted_group["prop_id"] = prop_id.values
    sorted_group["scores"] = scores
    sorted_group = sorted_group.sort_values(by= "scores", ascending=False)
    return sorted_group
# %% ordering by review
# train_review = train[['srch_id', 'prop_id', 'prop_review_score']]
# test = pd.read_csv("test_set_VU_DM.csv")
# print(test.isnull().sum())
# test_review = test[['srch_id', 'prop_id', 'prop_review_score']]
# result = byUserReview(test_review)

# final = result[['srch_id', 'prop_id']]
# final.to_csv('ByReview.csv', index=False))
# %%
test = pd.read_csv("test_set_VU_DM.csv")
test = test[X_train.columns]
test['prop_review_score'] = test['prop_review_score'].fillna(0)
test['prop_location_score2'] = test['prop_location_score2'].fillna(test['prop_location_score2'].mean())
test['orig_destination_distance'] = test['orig_destination_distance'].fillna(test['orig_destination_distance'].median())
print(test.isnull().sum())
# %%

lr = LogisticRegression()
lr.fit(X_train.drop(["srch_id", "prop_id"], axis=1, inplace=False), y_train)

#y_pred_lr = lr.predict_proba(X_test.head(100))
#scores = y_pred_lr[:,1]

result = optimal_recommendations(lr, test)
final = result[['srch_id', 'prop_id']]
final.to_csv('logistic.csv', index=False)

# # %%
# xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
# xgb.fit(X_train.drop(["srch_id", "prop_id"], axis=1, inplace=False), y_train)
    
# result = optimal_recommendations(xgb, test)
# final = result[['srch_id', 'prop_id']]
# final.to_csv('XgBoost.csv', index=False)