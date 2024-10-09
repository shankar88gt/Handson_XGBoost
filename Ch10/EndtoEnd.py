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

print(hot)




