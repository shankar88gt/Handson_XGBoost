import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(iris['data'],iris['target'],random_state=2)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb = XGBClassifier(booster='gbtree',objective='multi:softprob',max_depth=6,learning_rate=0.1,n_estimators=100,random_state=2,n_jobs=-1)
# multi:softprob - for multi class problem
#Learning rate - weightage of each tree
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('score:', score)

