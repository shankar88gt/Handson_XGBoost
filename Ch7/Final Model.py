# Tests with tuned parameters on Data size to check the effects

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 

import numpy as np

# BASE Model with label encoder
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv')
# to transform labels from 1,2 to 0,1
df['LABEL'] = df['LABEL'].replace(1,0)
df['LABEL'] = df['LABEL'].replace(2,1)

X = df.iloc[:,1:]
y = df.iloc[:,0]

weight = int(5050/37)

kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=2)

model=XGBClassifier(scale_pos_weight=weight,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')

print('Recall :',scores)
print('Recall Mean',scores.mean())
                    


