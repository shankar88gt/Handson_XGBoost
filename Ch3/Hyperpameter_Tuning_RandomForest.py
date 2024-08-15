""" The following are the hyperparameters for Random Forests
oob score - samples with replacement, test scores on those samples which are not selected
n_estimators 
warm_start - great for determining no of tress int he forest. from tress 100 to 200, it does not start from scratch 
bootstrap
verbose - not a help
depth
splits
    max features
    min_sample_split
    min_impurity_decrease
Leaves
    min_samples_leaves
    min_weight_fraction_leaf
"""
# warm start example
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
oob_scores = []

df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/census_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]

rf = RandomForestClassifier(n_estimators=50,warm_start=True,oob_score=True,n_jobs=-1,random_state=2)
rf.fit(X,y)
oob_scores.append(rf.oob_score_)

est = 0
estimators = [est]

for i in range(9):
    est += 50
    estimators.append(est)
    rf.set_params(n_estimators=est)
    rf.fit(X,y)
    oob_scores.append(rf.oob_score_)

plt.figure(figsize=(15,7))
plt.plot(estimators,oob_scores)
plt.xlabel('Estimators')
plt.ylabel('oob_scores')
plt.title("scores vs estimators")
plt.show()



