import pandas as pd
from ucimlrepo import fetch_ucirepo 
# fetch dataset 
census_income = fetch_ucirepo(id=20) 
  
# data (as pandas dataframes) 
X = census_income.data.features 
y = census_income.data.targets

X_Y = pd.concat([X, y], axis=1)

# removed all null values for simplicity
XY_fil = X_Y[X_Y['workclass'].notnull()]
XY_fil = XY_fil[XY_fil['native-country'].notnull()]
XY_fil['income'] = XY_fil['income'].apply(lambda x: '<=50K' if x == '<=50K.' else x)
XY_fil['income'] = XY_fil['income'].apply(lambda x: '>50K' if x == '>50K.' else x)
XY_fil1 = pd.get_dummies(XY_fil)
XY_fil1 = XY_fil1.drop('income_<=50K',axis = 1)
X = XY_fil1.iloc[:,:-1]
y = XY_fil1.iloc[:,-1]

# Simple linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

#############################################################
# Simple linear Reg
#############################################################

# fit & predict
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
print('log Reg accuracy', log_reg.score(X_test,y_test))

#############################################################
# XGboost
#############################################################

from xgboost import XGBClassifier
xg_clas = XGBClassifier()
xg_clas.fit(X_train,y_train)
print('XGBReg accuracy', xg_clas.score(X_test,y_test))

#############################################################
# Simple logistic Reg with cross validation
#############################################################
from sklearn.model_selection import cross_val_score
model = LogisticRegression()
scores = cross_val_score(model,X,y,cv=10)
print("Logist Accuracy:",np.round(scores,2))
print("Logistic Accuracy mean:, %0.2f" %(scores.mean()))

#############################################################
# XGboost with cross validation
#############################################################

from sklearn.model_selection import cross_val_score
model = XGBClassifier()
scores = cross_val_score(model,X,y,cv=10)
print("XGB Accuracy:",np.round(scores,2))
print("XGB Accuracy mean:, %0.2f" %(scores.mean()))
