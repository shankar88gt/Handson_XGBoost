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

print(accuracy)

"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix1  = confusion_matrix(y_pred,y_test)
display = ConfusionMatrixDisplay(confusion_matrix1).plot()
plt.show()

#visualize Decision Tree
tree.plot_tree(clf, filled=True)
plt.show()
"""