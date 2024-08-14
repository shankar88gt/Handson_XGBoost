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

#align Target variable
XY_fil['income'] = XY_fil['income'].apply(lambda x: '<=50K' if x == '<=50K.' else x)
XY_fil['income'] = XY_fil['income'].apply(lambda x: '>50K' if x == '>50K.' else x)
XY_fil1 = pd.get_dummies(XY_fil)
XY_fil1 = XY_fil1.drop('income_<=50K',axis = 1)

#Seperate Target vs Data
X = XY_fil1.iloc[:,:-1]
y = XY_fil1.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print("accuracy bfore Gridsearch: ", accuracy)

from sklearn.model_selection import GridSearchCV
params = {'max_depth' : [None,2,3,4,5,8,10,12,15,20] }

reg = tree.DecisionTreeClassifier(criterion='gini')
grid_reg = GridSearchCV(reg,params,scoring='accuracy',cv=5,n_jobs=-1)
grid_reg.fit(X_train,y_train)

best_params = grid_reg.best_params_
print("best Param:" , best_params)
print("best_score:", grid_reg.best_score_)

best_model = grid_reg.best_estimator_
y_pred = best_model.predict(X_test)
print("best Model:", accuracy_score(y_pred,y_test))

"""##########################################################
# Decision Tree Hyperparameters
##########################################################
->Max Depth
->min_samples_leaf
->max_leaf_nodes
->max_features
->min_samples_split
splitter
criterion
->min_impurity_decrease
min_weight_fraction_leaf
ccp_alpha
"""
