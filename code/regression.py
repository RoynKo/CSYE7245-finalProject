import pickle
import os
import gc
import pandas as pd
import numpy as np
from sklearn import linear_model
import time

start_time = time.time()
input_dir = os.path.join(os.pardir, 'data')
print('Loading data...')
rows_read = None
train_df = pd.read_csv(os.path.join(input_dir, 'sales_train.csv'), nrows=rows_read)
test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'), nrows=rows_read)
print(' Train data size is %d x %d'%(train_df.shape[0],train_df.shape[1]))
print('  Time elapsed %.0f sec'%(time.time()-start_time))


print('Pre-processing data...')
train_df.drop(['date','item_price'], axis=1, inplace=True)
train_agg = train_df.groupby(['date_block_num','shop_id','item_id'], as_index=False).agg(sum)
train_agg.columns = ['date_block_num','shop_id','item_id','item_cnt_month']
del train_df
gc.collect()


# normalize actuals to range of 0 to 20
train_agg['item_cnt_month'] = np.maximum(0, np.minimum(20, train_agg['item_cnt_month']))

# pivot
train_agg['date_block_num'] = 33 - train_agg['date_block_num'] # change it, so that it is 0 to 9
train_p = train_agg.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', aggfunc='sum', fill_value=0)
del train_agg
gc.collect()
data = np.array(train_p.values, dtype=np.int32)

# Create linear regression object
regr = linear_model.LinearRegression()

filename = 'finalized_model.sav'
pickle.dump(regr, open(filename, 'wb'))

# Train the model
ts = time.time()
x = data[:,1:10]
y = data[:,0]
regr.fit(x, y)
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))

# Make predictions
pred = regr.predict(x)

# The coefficients
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('MSRE: %.2f'%np.sqrt(((y-pred)*(y-pred)).mean()))

# prepare submission
p = regr.predict(data[:,0:9])
train_p['pred'] = p
train_p.drop('item_cnt_month',axis=1, inplace=True)
train_p.reset_index(level=['item_id', 'shop_id'], inplace=True)
s_df = pd.merge(test_df, train_p, how='left', on=['item_id', 'shop_id'])


# get sales by shop as % of average sales for all shops
ss = train_p.drop('item_id', axis=1).groupby('shop_id', as_index=False).agg(sum)
ss['pred'] = ss['pred'] / ss['pred'].mean() # 54 shops. range: 0.03 to 3.9

# get sales by item as % of average sales for all items
si = train_p.drop('shop_id', axis=1).groupby('item_id', as_index=False).agg(sum)
si['pred'] = si['pred'] / si['pred'].mean() # 11249 items. range: 0.03 to 90
s_df = pd.merge(s_df, ss, how='left', on='shop_id')
s_df = pd.merge(s_df, si, how='left', on='item_id')
s_df.columns = ['ID', 'shop_id','item_id', 'item_cnt_month', 'shop', 'item']
v = s_df['item_cnt_month'].mean()
s_df['pred2'] = v * s_df['shop'] * s_df['item'] * 0.225 # cut in X for new items
s_df['item_cnt_month'].fillna(s_df['pred2'], inplace=True)

# fill remaining missing with average by shop

s_df['pred3'] = v * s_df['shop'] * 0.225 # cut in X for new items
s_df['item_cnt_month'].fillna(s_df['pred3'], inplace=True)

# save
s_df.drop(['shop_id','item_id','shop','item','pred2','pred3'], axis=1, inplace=True)
s_df['item_cnt_month'] = np.maximum(0, np.minimum(20, s_df['item_cnt_month'])) # cap and floor
s_df.to_csv('submission_linear_regression.csv', index=False)
print('  Time elapsed %.0f sec'%(time.time()-start_time))
print( s_df.mean() )