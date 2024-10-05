# Oversampling 
# Warning - if the data is resampled before splitting into training & test sets. the recall score will be inflated 
#  same trainign samples present in training & test data - data leakage
#  

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv',nrows=400)
X = df.iloc[:,1:]
y = df.iloc[:,0]
# to transform labels from 1,2 to 0,1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y1 = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
# WITH IMBALANCED DATA, ACCURACY ISNT ENOUGH

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred,y_test))

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

#Oversampling
df_train = pd.merge(y_train,X_train,left_index=True,right_index=True)
new_df = pd.DataFrame(np.repeat(df_train[df_train['LABEL'] == 2].values,9,axis=0))
new_df.columns = df_train.columns
df_train_resample = pd.concat([df_train,new_df])
print(df_train_resample['LABEL'].value_counts())

X_train = df_train_resample.iloc[:,1:]
y_train = df_train_resample.iloc[:,0]    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()    
y1 = le.fit_transform(y_train)    

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y1)

y_pred = model.predict(X_test)       
ytest = le.fit_transform(y_test) 
print('accuracy',accuracy_score(ytest,y_pred))    
print(confusion_matrix(ytest,y_pred))
print(classification_report(ytest,y_pred))
#oversampling improves precision of 1 from 0.25 to 0.33 - slight improvement

