"""Early Stopping 

General Method to limit the number of training rounds in iterative ML
# eval_set, Eval_metric, Early_stopping_rounds

Instead of predetermining the number of training rounds, early stopping allows training to continue until n consecutive rounds fail to produce any gains, 
n is decided by user

its important to give the model sufficient time to fail. if the model stops too early , say after five rounds of no improvement. 
the model may miss general patters that it could pick up later. Boosting needs sufficient time to find intricate patterns within data

"""
import pandas as pd
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch6/heart_disease.csv')

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,random_state=2)

eval_set = [[X_test,y_test]]
eval_metric = 'error'

model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2,eval_metric=eval_metric)
model.fit(X_train,y_train,eval_set=eval_set)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Acuracy:", accuracy*100)

#Early Stopping rounds
model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2,eval_metric=eval_metric,early_stopping_rounds=10)
model.fit(X_train,y_train,eval_set=eval_set)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Acuracy:", accuracy*100)
# best results only after 3 trees; either patternt not learnt or dataset is small and 3 trees are sufficient

#through approach - n_estimators = 5000 & early stopping = 100
model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2,eval_metric=eval_metric,early_stopping_rounds=100,n_estimators=5000)
model.fit(X_train,y_train,eval_set=eval_set)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Acuracy:", accuracy*100)

# consider that early stopping is particularly useful for large datasets when its unclear how high you shd aim
