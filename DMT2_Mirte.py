import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('training_set_VU_DM.csv')

n, bins, patches = plt.hist(df.prop_country_id, 100, density = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Property country Id')
plt.title('Histogram of prop_country_id')
plt.show();

df.groupby('prop_country_id').size().nlargest(5)

n, bins, patches = plt.hist(df.visitor_location_country_id, 100, density = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Visitor location country Id')
plt.title('Histogram of visitor_location_country_id')
plt.savefig('Histogram visitor location country id')
plt.show();

df.groupby('visitor_location_country_id').size().nlargest(5)

us = df.loc[df['visitor_location_country_id'] == 219]
us = us.sample(frac=0.6, random_state=99)
del us['visitor_location_country_id']

us.isnull().sum()

cols_to_drop = ['date_time', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv', 'gross_bookings_usd', 'srch_id', 'prop_id']
us.drop(cols_to_drop, axis=1, inplace=True)

us['prop_location_score2'].fillna((us['prop_location_score2'].mean()), inplace=True)

us['orig_destination_distance'].fillna((us['orig_destination_distance'].median()), inplace=True)

sns.countplot(x='booking_bool',data=us, palette='hls')
plt.show();
us['booking_bool'].value_counts()

sns.countplot(x='click_bool',data=us, palette='hls')
plt.show();
us['click_bool'].value_counts()

n, bins, patches = plt.hist(us.srch_length_of_stay, 50, density = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Search length of stay')
plt.title('Histogram of search_length_of_stay')
plt.axis([0, 30, 0, 0.65])
plt.show();

us.groupby('srch_length_of_stay').size().nlargest(5)

n, bins, patches = plt.hist(us.srch_adults_count, 20, density = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Search adults count')
plt.title('Histogram of search_adults_count')
plt.show();

df.groupby('srch_adults_count').size().nlargest(5)

n, bins, patches = plt.hist(us.prop_starrating, 20, density = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Property star rating')
plt.title('Histogram of prop_star_rating')
plt.show();

us.groupby('prop_brand_bool').size()

us.groupby('srch_saturday_night_bool').size()

sns.set(style="ticks", palette="pastel")
ax = sns.boxplot(x="click_bool", y="price_usd", hue="click_bool", data=us)
ax.set_ylim([0, 200]);

us.groupby('click_bool')['price_usd'].describe()

click_indices = us[us.click_bool == 1].index
random_indices = np.random.choice(click_indices, len(us.loc[us.click_bool == 1]), replace=False)
click_sample = us.loc[random_indices]

not_click = us[us.click_bool == 0].index
random_indices = np.random.choice(not_click, sum(us['click_bool']), replace=False)
not_click_sample = us.loc[random_indices]

us_new = pd.concat([not_click_sample, click_sample], axis=0)

print("Percentage of not click impressions: ", len(us_new[us_new.click_bool == 0])/len(us_new))
print("Percentage of click impression: ", len(us_new[us_new.click_bool == 1])/len(us_new))
print("Total number of records in resampled data: ", len(us_new))
