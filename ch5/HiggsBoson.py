import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_higs = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch5/atlas-higgs-challenge-2014-v2.csv',nrows=250000)
print(df_higs.shape)
print(df_higs.columns.to_list())
df_higs.drop(columns=['Weight','KaggleSet'],inplace=True)
df_higs = df_higs.rename(columns={'KaggleWeight':'Weight'})

label_col = df_higs['Label']
df_higs.drop(columns=['Label'],inplace=True)
df_higs['Label'] = label_col

df_higs['Label'].replace(('s','b'),(1,0),inplace=True)
print(df_higs['Label'].value_counts())

print(df_higs.columns.to_list())

X = df_higs.iloc[:,1:31]
y = df_higs.iloc[:,-1]

# test weight for imbalanced data based on target in the test set
df_higs['test_weight'] = df_higs['Weight'] * 550000 / len(y)

s = np.sum(df_higs[df_higs['Label'] == 1 ]['test_weight'])
b = np.sum(df_higs[df_higs['Label'] == 0 ]['test_weight'])

print(b/s)

#Model
import xgboost as xgb
xgb_clf = xgb.DMatrix(X,y,missing=-999.0,weight=df_higs['test_weight'])
param = {}
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = b/s
param['eta'] = 0.1
param['max_depth'] = 6
param['eval_metric'] = 'auc'
plts = list(param.items())+[('eval_metric','ams@0.15')]
watchlist = [(xgb_clf,'train')]
num_round = 120
print('loading data and start to boost trees')
bst = xgb.train(plts,xgb_clf,num_round,watchlist)
bst.save_model('higs.model')
print('finish training')


















