import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',accuracy_score(y_pred,y_test))
# WITH IMBALANCED DATA, ACCURACY ISNT ENOUGH

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred,y_test))

#accuracy 0.89
#[[86  9]  -  86/88 - label 0 - 97.72%
# [ 2  3]] - 3/12 - label 1 - 25% 
# not good
# Precision = TP / TP + FP - 
# Recall = TP / TP + FN

#[0(TP),1(FN)
#1(FP),0(TN)]
#[X - Actual, y - Predicted]

print(classification_report(y_pred,y_test))

# With the above Confusion Matrix. our metric is Recall.

#Resampling -  imbalance Data
# Undersample or oversample the data

#Undersampling
#Intitial Data was undersampling -  400 /5087 entries
# Modify the above code to generalize

def xgb_clf(nrows):
    df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv',nrows=nrows)
    X = df.iloc[:,1:]
    y = df.iloc[:,0]    
    # to transform labels from 1,2 to 0,1
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()    
    y1 = le.fit_transform(y)    
    print('y1 Distribution - Training',pd.Series(y1).value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)
    print('y1 Distribution - Test Dataset',pd.Series(y_test).value_counts())
    model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)    
    print('accuracy',accuracy_score(y_pred,y_test))    
    print(confusion_matrix(y_pred,y_test))

#iteration 1 - full data set - 5000
print("Iteration 1")
xgb_clf(5087)
#iteration 2 - half dataset - 2500
print("Iteration 2")
xgb_clf(2543)
#iteration 3 - 25% - 1250
print("Iteration 3")
xgb_clf(1271)
#iteration 4 - 15% - 800
print("Iteration 4")
xgb_clf(800)
#iteration 5 - 7%  - 400
print("Iteration 5")
xgb_clf(400)
#iteration 6 - 3.5% - 200
print("Iteration 6")
xgb_clf(200)
#iteration 7 - Balanced - 37 - 50:50
print("Iteration 7")
xgb_clf(74)

"""
pay attension to 
    1) Precision of exoplanets starts
    2) Precision of non exoplanets starts
    3) Recall of exoplanets starts - this is critical
    4) Recall of non exoplanets starts

Iteration 1
y1 Distribution - Training     0    5050  1      37
y1 Distribution - Test Dataset 0    1261  1      11
accuracy 0.9913522012578616 : Recall : 0/11 
[[1261   11]
 [   0    0]]

Iteration 2
y1 Distribution - Training     0    2506  1      37
y1 Distribution - Test Dataset 0    630   1      6
accuracy 0.9905660377358491 : recall : 0/6
[[630   6]
 [  0   0]]


Iteration 3
y1 Distribution - Training     0    1234 1      37
y1 Distribution - Test Dataset 0    312  1      6
accuracy 0.9811320754716981 : recall : 0/6
[[312   6]
 [  0   0]]

Iteration 4
y1 Distribution - Training     0    763 1     37
y1 Distribution - Test Dataset 0    190 1     10
accuracy 0.955 : recall : 1/10
[[190   9]
 [  0   1]]

Iteration 5
y1 Distribution - Training     0    363 1     37
y1 Distribution - Test Dataset 0    88  1    12
accuracy 0.89 : recall : 3/12
[[86  9]
 [ 2  3]]


Iteration 6
y1 Distribution - Training     0    163 1     37
y1 Distribution - Test Dataset 0    37  1    13
accuracy 0.82 : recall : 4/13
[[37  9]
 [ 0  4]]


Iteration 7
y1 Distribution - Training     1    37 0    37
y1 Distribution - Test Dataset 1    11 0     8
accuracy 0.6842105263157895 : recall : 8/11 
[[5 3]
 [3 8]]
"""
