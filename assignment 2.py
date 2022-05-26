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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %%
train = pd.read_csv("training_set_VU_DM.csv")
# for col in train.columns:
#     pct_missing = np.mean(train[col].isnull())
#     print('{} - {}%'.format(col,round(pct_missing*100)))
train['prop_review_score'] = train['prop_review_score'].fillna(0)
train['prop_location_score2'] = train['prop_location_score2'].fillna(train['prop_location_score2'].median())
train['orig_destination_distance'] = train['orig_destination_distance'].fillna(train['orig_destination_distance'].median())
# %%
# count of null values in each column
print(train.isnull().sum())

# %%
train = train.dropna(axis="columns")
print(train.isnull().sum())

train = train.dropna(axis=0)
print(train.isnull().sum())
y_train = train["click_bool"]
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
test['prop_review_score'] = test['prop_review_score'].fillna(0)
test['prop_location_score2'] = test['prop_location_score2'].fillna(test['prop_location_score2'].median())
test['orig_destination_distance'] = test['orig_destination_distance'].fillna(test['orig_destination_distance'].median())
test = test[X_train.columns]
print(test.isnull().sum())
# %%

# lr = LogisticRegression()
# lr.fit(X_train.drop(["srch_id", "prop_id"], axis=1, inplace=False), y_train)

# result = optimal_recommendations(lr, test)
# final = result[['srch_id', 'prop_id']]
# final.to_csv('logistic.csv', index=False)

# %% testing performance
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)


#balancing test set did not work
# book_indices = X_train2[y_train2 == 1].index
# book_sample = X_train2.loc[book_indices]

# not_book = X_train2[y_train2 == 0].index
# random_indices = np.random.choice(not_book, sum(y_train2), replace=False)
# not_book_sample = X_train2.loc[random_indices]

# X_train2 = pd.concat([not_book_sample, book_sample], axis=0)
# y_train2 = y_train2[X_train2.index]
# print("Percentage of not click impressions: ", len(X_train2[y_train2 == 0])/len(X_train2))
# print("Percentage of click impression: ", len(X_train2[y_train2 == 1])/len(X_train2))
# print("Total number of records in resampled data: ", len(X_train2))

# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=1, stratify=y_train2)

xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth =5, n_estimators = 180)
xgb.fit(X_train2.drop(["srch_id", "prop_id"], axis=1, inplace=False), y_train2)
y_pred_xgb = xgb.predict(X_test2.drop(["srch_id", "prop_id"], axis=1, inplace=False))

acc_xgb = accuracy_score(y_test2, y_pred_xgb)
conf = confusion_matrix(y_test2, y_pred_xgb)
clf_report = classification_report(y_test2, y_pred_xgb)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_xgb}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")
# %%

xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
xgb.fit(X_train.drop(["srch_id", "prop_id"], axis=1, inplace=False), y_train)
    
result = optimal_recommendations(xgb, test)
final = result[['srch_id', 'prop_id']]
final.to_csv('XgBoostClick.csv', index=False)