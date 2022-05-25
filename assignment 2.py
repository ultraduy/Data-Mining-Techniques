# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %%
train = pd.read_csv("training_set_VU_DM.csv")

# %%
train.shape
#9917530 rows × 54 columns

# %%
train.dtypes

# %%
train.describe()

# %%
# count of null values in each column
print(train.isnull().sum())

# %%
train=train.dropna(axis=1,how="any")
print(train.isnull().sum())

# %%
train.shape

# %%
for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col,round(pct_missing*100)))

# %%
train.plot(x='date_time', y = 'price_usd', figsize = (20,5))
plt.xlabel('Date time')
plt.ylabel('Price in USD')
plt.title('Time Series of room price by date time of search')

# %%
import seaborn as sns
df = train.corr()
print(df)

train=train[['prop_starrating', 'prop_brand_bool', 'prop_location_score1',
       'prop_log_historical_price', 'price_usd', 'promotion_flag',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'random_bool']]

from sklearn.cluster import KMeans
data = train
n_cluster = range(1, 20)

kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize = (16, 8))
ax.plot(n_cluster, scores, color = 'orange')

plt.xlabel('clusters num')
plt.ylabel('score')
plt.title('Elbow curve for K-Means')
plt.show();

km = KMeans(n_clusters = 8)
kmean=km.fit(train)

y_kmeans = km.predict(train)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure(1, figsize = (7, 7))

ax = Axes3D(fig, rect = [0, 0, 0.95, 1], 
            elev = 48, azim = 134)

ax.scatter(train.iloc[:, 4:5],
           train.iloc[:, 7:8], 
           train.iloc[:, 11:12],
           c = km.labels_.astype(np.float), edgecolor = 'm')

ax.set_xlabel('USD')
ax.set_ylabel('srch_booking_window')
ax.set_zlabel('srch_saturday_night_bool')

plt.title('K Means', fontsize = 10)
# %%
def optimal_recommendations(model, test_data):
    result = pd.DataFrame(columns=["srch_id","prop_id","scores"])
    df_list = []
    grouped_data = data.groupby("srch_id")
    for group_name, group in tqdm(grouped_data):
        scores = model.predict(group.loc[:, ~group.columns.isin(["srch_id"])])
        sorted_group = sort_properties(group_name, scores, group["prop_id"])
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
    