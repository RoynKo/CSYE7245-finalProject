import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMRegressor
import time
import pickle

print("start")
DATA = '../data/'
sales = pd.read_csv(DATA+'sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
test = pd.read_csv(DATA+'test.csv')

df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

df_test = pd.merge(test, df, on=['item_id','shop_id'], how='left')
df_test = df_test.fillna(0)
df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)

TARGET = '2015-10'
y_train = df_test[TARGET]
X_train = df_test.drop(labels=[TARGET], axis=1)
X_test = df_test.drop(labels=['2013-01'],axis=1)

model=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=32,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.073,
        min_split_gain=0.0222415,
        min_child_weight=40)

ts= time.time()
model.fit(X_train, y_train,)
print('  For fitting: Time elapsed %.5f sec'%(time.time()-ts))

filename = 'finalized_model_lightGBM_raw.sav'
pickle.dump(model, open(filename, 'wb'))

y_pred = model.predict(X_test).clip(0., 20.)

preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])
preds.to_csv('submission_lightgbm_raw.csv',index_label='ID')


print("end")