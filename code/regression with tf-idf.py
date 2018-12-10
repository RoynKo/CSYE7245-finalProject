import time
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn

start_time = time.time()

input_dir = os.path.join(os.pardir, 'data')

rows_read = None
train_df = pd.read_csv(os.path.join(input_dir,'sales_train.csv'))
test_df = pd.read_csv(os.path.join(input_dir,'test.csv'))
items_df = pd.read_csv(os.path.join(input_dir,'items.csv'))
item_categories_df = pd.read_csv(os.path.join(input_dir,'item_categories.csv'))
shops_df = pd.read_csv(os.path.join(input_dir,'shops.csv'))



train_df = train_df.groupby([c for c in train_df.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train_df = train_df.rename(columns={'item_cnt_day':'item_cnt_month'})
train_df['item_cnt_month'] = np.maximum(0, np.minimum(20, train_df['item_cnt_month']))
train_df['date_block_num'] = 33 - train_df['date_block_num'] # change it, so that it is 0 to 9

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

#Monthly mean
shop_item_monthly_mean = train_df[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Features
train_df = pd.merge(train_df, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])

shop_item_prev_month = train_df[train_df['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})

train_df = pd.merge(train_df, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])

train_df.dropna(inplace= True, how = 'any')
train_df.drop('date', axis = 1,inplace=True)

regr = linear_model.LinearRegression()

x = train_df.iloc[:,2:]
y = train_df.iloc[:,0]
ts=time.time()
regr.fit(x, y)
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))

pred = regr.predict(x)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

print('MSRE: %.2f'%np.sqrt(((y-pred)*(y-pred)).mean()))

p = regr.predict(train_df.iloc[:,0:5])
train_df['pred'] = p


s_df = pd.merge(test_df, train_df, how='left', on=['item_id', 'shop_id'])


ss = train_df.drop('item_id', axis=1).groupby('shop_id', as_index=False).agg(sum)
ss['pred'] = ss['pred'] / ss['pred'].mean()

si = train_df.drop('shop_id', axis=1).groupby('item_id', as_index=False).agg(sum)
si['pred'] = si['pred'] / si['pred'].mean()
s_df = pd.merge(s_df, ss, how='left', on='shop_id')
s_df = pd.merge(s_df, si, how='left', on='item_id')

v = s_df['item_cnt_month'].mean()
s_df['pred2'] = v * 0.225
s_df['item_cnt_month'].fillna(s_df['pred2'], inplace=True)

s_df['pred3'] = v  * 0.225
s_df['item_cnt_month'].fillna(s_df['pred3'], inplace=True)

# save
s_df.drop(['shop_id','item_id','pred2','pred3'], axis=1, inplace=True)
s_df['item_cnt_month'] = np.maximum(0, np.minimum(20, s_df['item_cnt_month']))


sub = s_df[['ID', 'pred']]
sub[:].fillna(0, inplace=True)
sub1=sub.drop_duplicates(keep="first")
sub1.to_csv('submissionLR_with_tfidf.csv', index=False)

print('  Time elapsed %.0f sec'%(time.time()-start_time))