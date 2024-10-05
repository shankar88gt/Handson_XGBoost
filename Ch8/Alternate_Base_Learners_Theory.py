"""
BASIC THEORY of ALTERNATE BASE LEARNERS and OPTIONS AVAILABLE 

# Base learner - First model that XGboost uses to build the first model
# Decision Trees have energed as the preferred choice to XGboost; and also to RF & Extremly Randomized Trees
    # Https://scikit-learn.org/stable/modules/ensemble.html

1) gbtree        
2) gblinear
    DT are optimal for non linear data since real world data are often non linear
    There may be cases where linear model is ideal
    The General idea is the same; but each model in the ensemble is linear 
    Like Lasso & Ridge that add regularization terms, gblinear also add reg terms; multiple rounds of gblinear may be used to get back a single lasso reg
        https://github.com/dmlc/xgboost/issues/332 
3) Dart
    Dropout meets multiple Additive Regression Trees - 2015 
        https://proceedings.mlr.press/v38/korlakaivinayak15.pdf - UC Berkley
    Multiple additive Regression Trees ( MART ) as a successfull model that suffers too much dependency on earlier trees.
    instead of focusing on shrinkage; a standard penalization terms; they use dropout technique from Neural networks
    The dropout technique elimates nodes from each layer of learning from neural networks therby reducing overfitting. 
        basicaly the drop out technique slows down the learning process by elimating infrormation from each round
    Instead of summing the residuals from all prev trees to build a new model. it selects a random sample of previous tress and normalizzes the leaves by
        a scaling factor 1/k where k is the number of trees dropped
    XGboost implementation of DAART is similar to gbtree with additional parameters to accomodate dropouts
4) XGboost Random Forest
    Random forest may be implemented as base learners by setting num_parallel_trees > 1 and a class option within XGboost XGBRFRegressor & XGBRFClassifier
    Gradient boosting was designed to improve upon the errors of weak learners, not strong base learners like RF.
        nevertheless there may be fringe cases where RF base learners can be usefull
    XGBRF may be used as algorithms in thier own right; XGboost includes default hyperpara to deal with overfitting & their own methods for building individual trees
"""


