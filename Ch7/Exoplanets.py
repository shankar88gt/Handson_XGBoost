# This chapter we will do an end to end study of exoplanets 
# Exploratory analysis  
# approach Imbalance dataset
# Fine tuning the model 
import pandas as pd
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv',nrows=400)
#print(df.head())
print(df['LABEL'].value_counts())

import matplotlib.pyplot as plt
import numpy as np

X = df.iloc[:,1:]
y = df.iloc[:,0]

# to transform labels from 1,2 to 0,1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y1 = le.fit_transform(y)


def light_plot(index):
    y_vals = X.iloc[index]
    x_vals = np.arange(len(y_vals))
    plt.figure(figsize=(15,8))
    plt.xlabel('No of observations')
    plt.ylabel('light Flux')
    plt.title('light Plot' + str(index),size=15)
    plt.plot(x_vals,y_vals)
    plt.show()

# sample visualizations
#light_plot(0)
#light_plot(37)
#light_plot(1)
#light_plot(5)

#exploring
#print(df.info())
print('Null count',df.isnull().sum().sum())

#Quick & dirty Model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',accuracy_score(y_pred,y_test))
# WITH IMBALANCED DATA, ACCURACY ISNT ENOUGH

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred,y_test))

























