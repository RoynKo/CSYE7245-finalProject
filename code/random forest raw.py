import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.ensemble import RandomForestRegressor
import pickle

print('Loading data...')
train_df = pd.read_csv("../data/sales_train.csv")
test_df = pd.read_csv("../data/test.csv")
submission_df = pd.read_csv("../data/submission.csv")
items_df = pd.read_csv("../data/items.csv")
item_categories_df = pd.read_csv("../data/item_categories.csv")
shops_df = pd.read_csv("../data/shops.csv")

#turn data into monthly data
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year
train_df = train_df.drop(['date', 'item_price'], axis=1)
train_df = train_df.groupby([c for c in train_df.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train_df = train_df.rename(columns={'item_cnt_day':'item_cnt_month'})

#Monthly mean
shop_item_monthly_mean = train_df[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Features
train_df = pd.merge(train_df, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])


#Last Month : Oct 2015
shop_item_prev_month = train_df[train_df['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})


#Add the above previous month features
train_df = pd.merge(train_df, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
np.where(pd.isnull(train_df))
train_df = train_df.fillna(0.)

#Add Item, Category and Shop features
train_df = pd.merge(train_df, items_df, how='left', on='item_id')
train_df = pd.merge(train_df, item_categories_df, how='left', on=['item_category_id'])
train_df = pd.merge(train_df, shops_df, how='left', on=['shop_id'])

#Manipulate test data
test_df['month'] = 11
test_df['year'] = 2015
test_df['date_block_num'] = 34
test_df = pd.merge(test_df, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])

#Add previous month features
test_df = pd.merge(test_df, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
#Items features
test_df = pd.merge(test_df, items_df, how='left', on='item_id')
#Item Category features
test_df = pd.merge(test_df, item_categories_df, how='left', on='item_category_id')

#Shops features
test_df = pd.merge(test_df, shops_df, how='left', on='shop_id')
test_df = test_df.fillna(0.)
test_df['item_cnt_month'] = 0.


train_test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=True)
print("train_df.shape = {}, test_df.shape = {}, train_test_df.shape = {}".format(train_df.shape, test_df.shape, train_test_df.shape))
stores_hm = train_test_df.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
print("stores_hm.shape = {}".format(stores_hm.shape))


for c in ['shop_name', 'item_category_name', 'item_name']:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(list(train_df[c].unique()) + list(test_df[c].unique()))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test_df[c].astype(str))
    print(c)
feature_list = [c for c in train_df.columns if c not in 'item_cnt_month']

#Validation hold out month is 33 if necessary
x1 = train_df[train_df['date_block_num'] < 33]
y1 = np.log1p(x1['item_cnt_month'].clip(0., 20.))
x1 = x1[feature_list]
x2 = train_df[train_df['date_block_num'] == 33]
y2 = np.log1p(x2['item_cnt_month'].clip(0., 20.))
x2 = x2[feature_list]

print("start training")
rf = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)

ts=time.time()
#Full train
rf.fit(train_df[feature_list], train_df['item_cnt_month'].clip(0., 20.))

filename = 'finalized_model_random_forest.sav'
pickle.dump(rf, open(filename, 'wb'))

print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))
print("Accuracy on training data without considering variable importances:{}".format(round(rf.score(train_df[feature_list], train_df['item_cnt_month'].clip(0., 20.))*100, 2)))

#predict
test_df['item_cnt_month'] = rf.predict(test_df[feature_list]).clip(0., 20.)
#create submission file
test_df[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)

# Extract the six most important features
rf_most_important = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)
important_features = ['item_cnt_month_mean', 'date_block_num', 'item_cnt_prev_month', 'item_id', 'item_name', 'year']
ts=time.time()
#Full train
rf_most_important.fit(train_df[important_features], train_df['item_cnt_month'].clip(0., 20.))
filename = 'finalized_model_random_forest_mi.sav'
pickle.dump(rf, open(filename, 'wb'))

print("Accuracy on training data considering variable importances:{}".format(round(rf_most_important.score(train_df[important_features], train_df['item_cnt_month'].clip(0., 20.))*100, 2)))
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))
#predict
test_df['item_cnt_month'] = rf_most_important.predict(test_df[important_features]).clip(0., 20.)
#create submission file
test_df[['ID', 'item_cnt_month']].to_csv('submission_variable_importance.csv', index=False)