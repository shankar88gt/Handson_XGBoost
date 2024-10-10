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
final_sparse_matrx = hstack(hot,cold)

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
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
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
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # get the cat columns list
        cat_columns = df.columns[X.dtypes == object].to_list()
        # filter all numerica data
        Num_X = X.select_dtypes(exclude=['object'])
        # one hot encoding of categorical columns
        hot = ohe.fit_transform(X[cat_columns])
        # convert to sparse Matrix
        cold = csr_matrix(Num_X)
        # horizontal stack cat & num data which are both converted to sparse matrix
        final_sparse_matrx = hstack(hot,cold)
        # Compressed Sparse Row ( CSR ) - Sparse data stored in row format
        final_CSR = pd.DataFrame(final_sparse_matrx.tocsr())
        return final_CSR
    
sm = SparseMatrix().fit_transform(nvi)
print(sm)

# to convert back to dense matrix
# sm_df = pd.dataframe(sm.toarray())

# Preprocessing Pipeline 
# Add null imputer & sparmatrix transformers to pipeline
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=2)

from sklearn.pipeline import Pipeline
data_pipeline = Pipeline([('null imputer',NullValueImputer()),('sparse',SparseMatrix())])

X_train_tranformed = data_pipeline.fit_transform(X_train)

# Model 













