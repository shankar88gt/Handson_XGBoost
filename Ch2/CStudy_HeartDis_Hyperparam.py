import pandas as pd
#from ucimlrepo import fetch_ucirepo 
# fetch dataset 
#heart_disease = fetch_ucirepo(id=45)  
# data (as pandas dataframes) 
#X = heart_disease.data.features 
#y = heart_disease.data.targets 

df_heart = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch2/heart_disease.csv')
X= df_heart.iloc[:,:-1]
y= df_heart.iloc[:,-1]

#combine Data & target for preprocessing
XY = pd.concat([X, y], axis=1)

#remove nulls
XY = XY[XY['ca'].notnull()]
XY = XY[XY['thal'].notnull()]

#One hot encoding
#XY = pd.get_dummies(XY)

#Seperate Target vs Data
X = XY.iloc[:,:-1]
y = XY.iloc[:,-1]

#Base Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)

print("before Hyper tuning :",accuracy)

#Hyperparameters tuning
from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [None,2,3,4,5,8,10,12,15,20],
          'min_samples_split':[2,3,4,7,9,10,13,15,17],
          'min_samples_leaf':[1,2,3,5,7,10],
          'max_leaf_nodes':[10,15,20,25,30,35,40]          
        }

reg = tree.DecisionTreeClassifier()
grid_reg = GridSearchCV(reg,params,scoring='accuracy',cv=5,n_jobs=-1)
grid_reg.fit(X_train,y_train)

best_params = grid_reg.best_params_
print("best Param:" , best_params)
print("best_score:", grid_reg.best_score_)

best_model = grid_reg.best_estimator_
y_pred = best_model.predict(X_test)
print("Test score on Best Model:", accuracy_score(y_pred,y_test))

#feature importance
feature_dict = dict(zip(X.columns,best_model.feature_importances_))
import operator
print(sorted(feature_dict.items(),key=operator.itemgetter(1),reverse=True))
