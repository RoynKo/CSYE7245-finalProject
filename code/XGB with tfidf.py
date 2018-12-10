import numpy as np
import pandas as pd
import sklearn
import nltk
import time
from xgboost import XGBRegressor
import pickle


print("Start preparing")
train_df = pd.read_csv("../data/sales_train.csv")
test_df = pd.read_csv("../data/test.csv")
submission_df = pd.read_csv("../data/submission.csv")
items_df = pd.read_csv("../data/items.csv")
item_categories_df = pd.read_csv("../data/item_categories.csv")
shops_df = pd.read_csv("../data/shops.csv")

feature_count = 25
items_df['item_name_length'] = items_df['item_name'].map(lambda x : len(x)) #Length of each item_name(including punctuation in the item_name)
items_df['item_name_word_count'] = items_df['item_name'].map(lambda x : len(x.split(' '))) #Number of words/group of characters seperated by a whitespace
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count) #tfidf = term frequency inverse document frequency
items_df_item_name_text_features = pd.DataFrame(tfidf.fit_transform(items_df['item_name']).toarray())
print("Shape of items_df_item_name_text_features : {}".format(items_df_item_name_text_features.shape))
cols = items_df_item_name_text_features.columns
for idx in range(feature_count):
    items_df['item_name_tfidf_' + str(idx)] = items_df_item_name_text_features[cols[idx]]


feature_count = 25
item_categories_df['item_categories_name_length'] = item_categories_df['item_category_name'].map(lambda x : len(x)) #Length of each item_category_name(including punctuation in the item_category_name)
item_categories_df['item_categories_name_word_count'] = item_categories_df['item_category_name'].map(lambda x : len(x.split(' '))) #Number of words/group of characters seperated by a whitespace
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count) #tfidf = term frequency inverse document frequency
item_categories_df_item_category_name_text_features = pd.DataFrame(tfidf.fit_transform(item_categories_df['item_category_name']).toarray())
cols = item_categories_df_item_category_name_text_features.columns
for idx in range(feature_count):
    item_categories_df['item_category_name_tfidf_' + str(idx)] = item_categories_df_item_category_name_text_features[cols[idx]]


feature_count = 25
shops_df['shop_name_length'] = shops_df['shop_name'].map(lambda x : len(x)) #Length of each shop_name(including punctuation in the shop_name)
shops_df['shop_name_word_count'] = shops_df['shop_name'].map(lambda x : len(x.split(' '))) #Number of words/group of characters seperated by a whitespace
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count) #tfidf = term frequency inverse document frequency
shops_df_shop_name_text_features = pd.DataFrame(tfidf.fit_transform(shops_df['shop_name']).toarray())
cols = shops_df_shop_name_text_features.columns
for idx in range(feature_count):
    shops_df['shop_name_tfidf_' + str(idx)] = shops_df_shop_name_text_features[cols[idx]]

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



train_test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
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
#Validation hold out month is 33
x1 = train_df[train_df['date_block_num'] < 33]
y1 = np.log1p(x1['item_cnt_month'].clip(0., 20.))
x1 = x1[feature_list]
x2 = train_df[train_df['date_block_num'] == 33]
y2 = np.log1p(x2['item_cnt_month'].clip(0., 20.))
x2 = x2[feature_list]

print("Preparation finished")

##XGB

##X_test = test_df.drop(labels=['2013-01'],axis=1)
print("start xgb")
ts = time.time()
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    x1, 
    y1, 
    eval_metric="rmse", 
    eval_set=[(x1, y1), (x2, y2)], 
    verbose=True, 
    early_stopping_rounds = 5)
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))

filename = 'finalized_model_XGB_tfidf.sav'
pickle.dump(model, open(filename, 'wb'))

test_df['item_cnt_month'] = model.predict(test_df[feature_list]).clip(0., 20.)
test_df[['ID', 'item_cnt_month']].to_csv('submissionXGB.csv', index=False)
print('  For predicting: Time elapsed %.5f sec'%(time.time()-ts))