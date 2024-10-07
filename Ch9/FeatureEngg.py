# Feature Engineering 

#  The question is how muh feature engineering shd you implement?
# Whats covered here are basics; refer to AAAMLP feature Engg nd combine approaches for Feature Engineering.

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier, XGBRFClassifier, XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch9/cab_rides.csv',nrows=10000)
print(df.head())

# Handle Null values in Target & other columns
print(df.info())
print(df[df.isna().any(axis=1)])
df.dropna(inplace=True)

# Timestamp Columns - Convert them to usefull info
df['date'] = pd.to_datetime(df['time_stamp'])*(10**6) # some problem with convertion in the data and use this to correct it.

import datetime as dt
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.day_of_week

def weekend(row):
    if row['dayofweek'] in [5,6]:
        return 1
    else:
        return 0

df['weekend'] = df.apply(weekend,axis=1)

def rush_hour(row):
    if (row['hour'] in [6,7,8,9,15,16,17,18]) & (row['weekend'] == 0):
        return 1
    else:
        return 0

df['rush_hour'] = df.apply(rush_hour,axis=1)

# Engineering Frequency Columns

print(df['cab_type'].value_counts())
df['cab_freq'] = df.groupby('cab_type')['cab_type'].transform('count')
df['cab_freq'] = df['cab_freq'] / len(df)

# Target Encoding
# There is data leakage while using target values; additional regularization technique are required
# Be carefull of DATA Leakage


#Steps: pip install --upgrade category_encoders

from category_encoders.target_encoder import TargetEncoder
encoder = TargetEncoder()
df['cab_type_mean'] = encoder.fit_transform(df['cab_type'], df['Price'])

#https://www.kaggle.com/code/vprokopev/mean-likelihood-encodings-a-comprehensive-study

# We may Engineer thousands of new columns 
# if you have too many engineered columns. you can select using the feature_importance_ 






