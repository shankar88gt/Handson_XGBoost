"""
XGBoost Hyperparameters list

eta [default=0.3, alias: learning_rate]
    Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, 
    and eta shrinks the feature weights to make the boosting process more conservative.
    range: [0,1]

gamma [default=0, alias: min_split_loss]
    Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. 
    Note that a tree where no splits were made might still contain a single terminal node with a non-zero score.
    range: [0,∞]
    
max_depth [default=6]
    Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. 
    Beware that XGBoost aggressively consumes memory when training a deep tree. exact tree method requires non-zero value.
    range: [0,∞]
    
min_child_weight [default=1]
    Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, 
    then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. 
    The larger min_child_weight is, the more conservative the algorithm will be.
    range: [0,∞]

max_delta_step [default=0]
    Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, 
    it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. 
    Set it to value of 1-10 might help control the update.
    range: [0,∞]

subsample [default=1]
    Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
    range: (0,1]
    
sampling_method [default= uniform]
    The method to use to sample the training instances.
    uniform: each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
    gradient_based: the selection probability for each training instance is proportional to the regularized absolute value of gradients (more specifically, ). subsample may be set to as low as 0.1 without loss of model accuracy. Note that this sampling method is only supported when tree_method is set to hist and the device is cuda; other tree methods only support uniform sampling.
    

colsample_bytree, colsample_bylevel, colsample_bynode [default=1]
    This is a family of parameters for subsampling of columns.
    All colsample_by* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.
    colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    colsample_bylevel is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. 
        Columns are subsampled from the set of columns chosen for the current tree.
    colsample_bynode is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. 
        Columns are subsampled from the set of columns chosen for the current level. This is not supported by the exact tree method.
    colsample_by* parameters work cumulatively. For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5} 
        with 64 features will leave 8 features to choose from at each split.


Using the Python or the R package, one can set the feature_weights for DMatrix to define the probability of each feature being selected when using column sampling. 
    There’s a similar parameter for fit method in sklearn interface.

lambda [default=1, alias: reg_lambda]
    L2 regularization term on weights. Increasing this value will make model more conservative.
    range: [0, ]

alpha [default=0, alias: reg_alpha]
    L1 regularization term on weights. Increasing this value will make model more conservative.
    range: [0, ]
    tree_method string [default= auto]

The tree construction algorithm used in XGBoost. See description in the reference paper and Tree Methods.
    Choices: auto, exact, approx, hist, this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
    auto: Same as the hist tree method.
    exact: Exact greedy algorithm. Enumerates all split candidates.
    approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
    hist: Faster histogram optimized approximate greedy algorithm.

scale_pos_weight [default=1]
    Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances). See Parameters Tuning for more discussion. Also, see Higgs Kaggle competition demo for examples: R, py1, py2, py3.
    updater

A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updaters exist:
    grow_colmaker: non-distributed column-based construction of trees.
    grow_histmaker: distributed tree construction with row-based data splitting based on global proposal of histogram counting.
    grow_quantile_histmaker: Grow tree using quantized histogram.
    grow_gpu_hist: Enabled when tree_method is set to hist along with device=cuda.
    grow_gpu_approx: Enabled when tree_method is set to approx along with device=cuda.

sync: synchronizes trees in all distributed nodes.
    refresh: refreshes tree’s statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
    prune: prunes the splits where loss < min_split_loss (or gamma) and nodes that have depth greater than max_depth.


refresh_leaf [default=1]
    This is a parameter of the refresh updater. When this flag is 1, tree leafs as well as tree nodes’ stats are updated. When it is 0, only node stats are updated.

process_type [default= default]
    A type of boosting process to run.
    Choices: default, update
        default: The normal boosting process which creates new trees.
        update: Starts from an existing model and only updates its trees. In each boosting iteration, a tree from the initial model is taken, 
    a specified sequence of updaters is run for that tree, and a modified tree is added to the new model. 
    The new model would have either the same or smaller number of trees, depending on the number of boosting iterations performed. Currently, 
    the following built-in updaters could be meaningfully used with this process type: refresh, prune. With process_type=update, one cannot use updaters that create new trees.

grow_policy [default= depthwise]
    Controls a way new nodes are added to the tree.
    Currently supported only if tree_method is set to hist or approx.
    Choices: depthwise, lossguide
    depthwise: split at nodes closest to the root.
    lossguide: split at nodes with highest loss change.

max_leaves [default=0]
    Maximum number of nodes to be added. Not used by exact tree method.

max_bin, [default=256]
    Only used if tree_method is set to hist or approx.
    Maximum number of discrete bins to bucket continuous features.
    Increasing this number improves the optimality of splits at the cost of higher computation time.

num_parallel_tree, [default=1]
    Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.    

"""

import pandas as pd
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch6/heart_disease.csv')
#print(df.info())

from xgboost import XGBClassifier

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2)

from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(model,X,y,cv=5)
print('Base Accuracy',np.round(scores,2))
print('Base Accuracy mean', scores.mean())

#using stratified Kfold
#Iteration 1
from HyperParametrs_Grid_RandmCV_Combined import grid_search
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,random_state=2,shuffle=True)
grid_search(kfold,X,y,params={'n_estimators':[100,200,300,500,800]})

#Iteration 2
grid_search(kfold,X,y,params={'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5]})

#Iteration 3
grid_search(kfold,X,y,params={'max_depth':[2,3,5,6,8,10]})

#Iteration 4
grid_search(kfold,X,y,params={'gamma':[0,0.1,0.5,1,2,5,8]})

#Iteration 5
grid_search(kfold,X,y,params={'min_child_weight':[1,2,3,4,5]})

#Iteration 6
grid_search(kfold,X,y,params={'subsample':[0.5,0.6,0.7,0.8,0.9,1]})

#Iteration 7
grid_search(kfold,X,y,params={'colsample_bytree':[0.5,0.7,0.8,0.9,1]})