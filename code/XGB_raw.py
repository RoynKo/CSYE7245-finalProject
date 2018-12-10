import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pickle

print('Loading data...')
items = pd.read_csv('../data/items.csv')
train = pd.read_csv('../data/sales_train.csv')
test = pd.read_csv('../data/test.csv')
item_category = pd.read_csv('../data/item_categories.csv')
shops = pd.read_csv('../data/shops.csv')

def drop_duplicate(data, subset):
    data.drop_duplicates(subset,keep='first', inplace=True) 
    data.reset_index(drop=True, inplace=True)
    after = data.shape[0]


subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']

drop_duplicate(train, subset = subset)

train = train[(train.item_price > 0) & (train.item_price < 300000)]

l = list(item_category.item_category_name)
l_cat = l
for ind in range(1,8):
    l_cat[ind] = 'Access'
for ind in range(10,18):
    l_cat[ind] = 'Consoles'
for ind in range(18,25):
    l_cat[ind] = 'Consoles Games'
for ind in range(26,28):
    l_cat[ind] = 'phone games'
for ind in range(28,32):
    l_cat[ind] = 'CD games'
for ind in range(32,37):
    l_cat[ind] = 'Card'
for ind in range(37,43):
    l_cat[ind] = 'Movie'
for ind in range(43,55):
    l_cat[ind] = 'Books'
for ind in range(55,61):
    l_cat[ind] = 'Music'
for ind in range(61,73):
    l_cat[ind] = 'Gifts'
for ind in range(73,79):
    l_cat[ind] = 'Soft'


item_category['cats'] = l_cat
train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")

p_df = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

train_cleaned_df = p_df.reset_index()
train_cleaned_df['shop_id']= train_cleaned_df.shop_id.astype('str')
train_cleaned_df['item_id']= train_cleaned_df.item_id.astype('str')
item_to_cat_df = items.merge(item_category[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]
item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')
train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")

number = preprocessing.LabelEncoder()
train_cleaned_df[['cats']] = number.fit_transform(train_cleaned_df.cats)
train_cleaned_df = train_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]

param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

progress = dict()
xgbtrain = xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values, train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values)
watchlist = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)

filename = 'finalized_model_xgb_raw.sav'
pickle.dump(bst, open(filename, 'wb'))

preds = bst.predict(xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values))
rmse = np.sqrt(mean_squared_error(preds,train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values))
apply_df = test
apply_df['shop_id']= apply_df.shop_id.astype('str')
apply_df['item_id']= apply_df.item_id.astype('str')

apply_df = test.merge(train_cleaned_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

d = dict(zip(apply_df.columns[4:],list(np.array(list(apply_df.columns[4:])) - 1)))

apply_df  = apply_df.rename(d, axis = 1)

ts=time.time()
preds = bst.predict(xgb.DMatrix(apply_df.iloc[:, (apply_df.columns != 'ID') & (apply_df.columns != -1)].values))
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))

preds = list(map(lambda x: min(20,max(x,0)), list(preds)))
sub_df = pd.DataFrame({'ID':apply_df.ID,'item_cnt_month': preds })

sub_df.to_csv('Submission_Predict_Salesxgb.csv',index=False)

print("finish")