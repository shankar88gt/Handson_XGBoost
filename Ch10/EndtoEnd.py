# End to End Project with Transformers 

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch10/student-por.csv', sep=';')
print(df.head())

print(df.isna().sum())
# Null in sex, age & gardian

# Handling Null
# XGboost required numerical input so the missing hyperparameter cannot be directly applied to categorial columns as of 2020

df['sex'] = df['sex'].fillna(df['sex'].mode())
df['guardian'] = df['guardian'].fillna(df['guardian'].mode())

print(df.isna().sum())

#One Hot encoding
# 2 approachs - a) pd.get_dummies / scikit learn onehotencoder
# Pd.get_dummies - computationally expensive & does not translate particularly well to scikit learn pipelines
# OneHotEncoder - sparse matrix, works well with scikit learn pipelines
# 
cat_columns = df.columns[df.dtypes == object].to_list()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
hot = ohe.fit_transform(df[cat_columns])

hot_df = pd.DataFrame(hot.toarray())
print(hot_df.head())

# print(hot)  - the above method is sparse matrix, uncomment and verify

#Combining one hot encoding matrix & numerical columns

# Only numerical columns
cold_df = df.select_dtypes(exclude=['object'])
print(cold_df.head())

# for combining 2 options
# 1) convert cat columns dataset to dense matrix and combine
# 2) comvert numerical to sparse matrix - we will go with this one as it will help us int he future

from scipy.sparse import csr_matrix
cold = csr_matrix(cold_df)

from scipy.sparse import hstack
final_sparse_matrx = hstack((hot,cold))

final_df = pd.DataFrame(final_sparse_matrx.toarray())
print(final_df.head())

#Transformers
# Scikit learn transformers works by using a fit methos & a transforms method
# fit learns the parameters ; transforms method applies these parameters to the data
# can be combined into one single method - fit_transform

# scikit learn has many transformers - StandardScaler (standerdize) , normalizer (normalize), simpleimputer for null values
# its worth creating your own transformers  - TransformerMixin as your superclass

#template
"""
class Yourclass(TransformerMixin):
    def __init__(self):
        None
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        #insert code to transform X
        return X
"""

from sklearn.base import TransformerMixin
class NullValueImputer(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for column in X.columns.to_list():
            if column in X.columns[X.dtypes == object].to_list():
                X[column] = X[column].fillna(X[column].mode())
            else:
                X[column] = X[column].fillna(-999.0)
        return X

# df - dataset after reading the csv file - see line 1
nvi = NullValueImputer().fit_transform(df)
print(nvi.head())

# transformers for One hot encoding

class SparseMatrix(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # get the cat columns list
        cat_columns = X.columns[X.dtypes == object].to_list()
        # filter all numerica data
        Num_X = X.select_dtypes(exclude=['object'])
        # one hot encoding of categorical columns
        hot = ohe.fit_transform(X[cat_columns])
        # convert to sparse Matrix        
        cold = csr_matrix(Num_X)
        # horizontal stack cat & num data which are both converted to sparse matrix
        final_sparse_matrx = hstack((hot,cold))
        # Compressed Sparse Row ( CSR ) - Sparse data stored in row format
        final_CSR = final_sparse_matrx.tocsr()
        return final_CSR

# to Test    
#sm = SparseMatrix().fit_transform(nvi)
#print(sm)




# to convert back to dense matrix
# sm_df = pd.DataFrame(sm.toarray())
# print(sm_df.head())

# Preprocessing Pipeline 
# Add null imputer & sparmatrix transformers to pipeline
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch10/student-por.csv', sep=';')

X = df.iloc[:, :-3]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=2)

from sklearn.pipeline import Pipeline
data_pipeline = Pipeline([('null imputer',NullValueImputer()),('sparse',SparseMatrix())])

X_train_tranformed = data_pipeline.fit_transform(X_train)

# Model 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor

print(y_train.value_counts())
# Since the target column is ordinal, Regression is preferable even though outputs are limited
# final results may be rounded to give the final predictions
# Evaluation metric is RMSE

kfold = KFold(n_splits=5,shuffle=True,random_state=2)

# cross validation function that returns a RMSE 
def cross_val(model):
    scores = cross_val_score(model,X_train_tranformed,y_train,scoring='neg_root_mean_squared_error',cv=kfold)
    rmse = (-scores.mean())
    return rmse

print(cross_val(XGBRegressor(missing=-999.0)))

# Analysis of results
# RMSE - 2.97,  2.97 out of 19 possibilities indicates that the grades are within couple of points of accuracy 
# This accurace withi 1 letter grade accuracy
# In industry you may even include a confidence level interval statistics 
# To be researched


# hyper parameters tuning
X_train_2, X_test_2,y_train_2,y_test_2 = train_test_split(X_train_tranformed,y_train,random_state=2)

def n_estimators(model):
    eval_set = [(X_test_2,y_test_2)]
    eval_metric = 'rmse'
    model.fit(X_train_2,y_train_2,eval_metric=eval_metric,eval_set=eval_set,early_stopping_rounds=100)
    y_pred = model.predict(X_test_2)
    rmse = MSE(y_test_2,y_pred)**0.5
    return rmse

print(n_estimators(XGBRegressor(n_estimators=5000,missing=-999.0)))

# grid Search - fine tuning hyper parameters
def grid_search(params,reg=XGBRegressor(missing=-999.0)):
    grid_reg = GridSearchCV(reg,params,scoring='neg_mean_squared_error',cv=kfold)
    grid_reg.fit(X_train_tranformed,y_train)
    best_params = grid_reg.best_params_
    best_score = np.sqrt(-grid_reg.best_score_)
    print('Best Param', best_params)
    print('Best Score', best_score)

#iteration 1
grid_search(params={'max_depth':[1,2,3,4,6,7,8],
                    'n_estimators':[15,20,25,31]
                    })
#iteration 2
grid_search(params={'max_depth':[1,2,3],
                    'min_child_weight':[1,2,3,4,5],
                    'n_estimators':[31]
                    })
#iteration 3
grid_search(params={'max_depth':[2,3],
                    'min_child_weight':[2,3,4],
                    'subsample':[0.5,0.6,0.7,0.8],
                    'n_estimators':[31,50]
                    })
#iteration 4
grid_search(params={'max_depth':[1,2],
                    'min_child_weight':[1,2,3,4],
                    'subsample':[0.6,0.7,0.8],
                    'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
                    'n_estimators':[40,50]
                    })
#iteration 5
grid_search(params={'max_depth':[1],
                    'min_child_weight':[3],
                    'subsample':[0.7,0.8],
                    'colsample_bytree':[0.6,0.7,0.8,0.9],
                    'colsample_bylevel':[0.6,0.7,0.8,0.9],
                    'colsample_bynode':[0.6,0.7,0.8,0.9],
                    'n_estimators':[50]
                    })


# Post Hypertuning - Test the Model
# The transformation wasnt applied to test data
# use the pipeline to to transform the test data

X_test_tranformed = data_pipeline.fit_transform(X_test)

model = XGBRegressor(max_depth=2,min_child_weight=3,subsample=0.8,colsample_bytree=0.6,colsample_bylevel=0.8,colsample_bynode=0.7,n_estimators=50)
model.fit(X_test_tranformed,y_train)
y_pred = model.predict(X_test_tranformed)
rmse = MSE(y_test,y_pred)**0.5
print('RMSE',rmse)

# the score can be little high on the testset
# Next steps are
# Return to Hyperparameter tuning
# Keep the model as is
# Make a quick adjustment based on Hyperparameter knowledge

#Final adjustment based on Knowledge

model = XGBRegressor(max_depth=1,min_child_weight=5,subsample=0.6,colsample_bytree=0.9,colsample_bylevel=0.9,colsample_bynode=0.8,n_estimators=50)
model.fit(X_test_tranformed,y_train)
y_pred = model.predict(X_test_tranformed)
rmse = MSE(y_test,y_pred)**0.5
print('RMSE',rmse)

#building a ML pipeline
# Add model to your pipeline

full_pipeline = Pipeline([('null imputer',NullValueImputer()),('sparse',SparseMatrix()), 
                          ('xgb',XGBRegressor(max_depth=1,min_child_weight=5,subsample=0.6,colsample_bytree=0.9,colsample_bylevel=0.9,colsample_bynode=0.8,n_estimators=50))])

full_pipeline.fit(X,y)
new_data = X_test
full_pipeline.predict(new_data)

# In Reality, When New data comes in, it can be concatenated with Previous data and placed thru the sam pipeline for stronger Model
# IF you want to Make prediction on only one row of data, if u run a single row of data the pipeline, since we have used sparse matrix; 
# it will not have the correct columns if not all columns are populated. this results in Mismatch error
# a simple solution is to concat with additional rows and then run thru the pipeline




