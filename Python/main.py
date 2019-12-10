# %%
# Load library
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
from joblib import load, dump

# Hardware differences
if 'KAGGLE_WORKING_DIR' in os.environ:
    # On Kaggle
    print('Kaggle setup')
elif 'COLAB_GPU' in os.environ:
    # On Google Colaboratory
    print('Colab setup')
    from google.colab import drive
    drive_path = '/content/drive' 
    drive.mount(drive_path)
    work_path = drive_path + '/My Drive/kaggle/Housing Prices Competition for Kaggle Learn Users'
    lib_path = drive_path + '/My Drive/python/My Utility'
    os.chdir(work_path)
    sys.path.append(lib_path)
else:
    # On Local computer
    print('Local setup')
    # Local settings
    # default windows_user_profile = os.environ['USERPROFILE']
    # default work_path = windows_user_profile + '/JupyterNotebook/kaggle/Housing Prices Competition for Kaggle Learn Users'
    work_path = 'D:/Projects/JupyterNotebook/kaggle/Housing Prices Competition for Kaggle Learn Users'
    lib_path = 'D:/Projects/Python/My Utility'
    os.chdir(work_path)
    sys.path.append(lib_path)

# Load my utilities
import analyze as anlz
import refine as refn
import simulation as sim
from simulation import grid_search, random_search
import utility as utl

# Look around the current folder
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

del([os, sys, work_path, lib_path])

# %%
# Reload library
import importlib

importlib.reload(anlz)
importlib.reload(refn)
importlib.reload(sim)
importlib.reload(utl)

del(importlib)

# %%
# Settings
path = '../input/home-data-for-ml-course'

# Search hyperparameter or predict with best estimator
search_hyperparameter = False

# Parameter of cross validation
VERBOSE_ = 1
N_JOBS_ = -1
CV_ = 4
ERROR_SCORE_ = 0.0

# Pandas settings. Show all rows and columns
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 200

# Prediction target
target = 'SalePrice'

# %%
# Read files
# Read csv data and store the data to pandas DataFrame
train = pd.read_csv(f'{path}/train.csv', index_col=0)
test = pd.read_csv(f'{path}/test.csv', index_col=0)

# My route
# Add extensional data
extension = pd.read_csv(f'{path}-extension/extension.csv', index_col=0)
predicted_test = pd.read_csv(f'{path}-extension/predicted_test.csv', index_col=0)

# Keep ID for submission because it will be removed.
Id = test.index.values

# Keep dtypes
classified_features = anlz.classify_samples_by_dtypes(train.drop(columns=target))

# %%
# Outliers
# Treat outliers
train, test = refn.treat_outliers(train, test)

# %%
# Combine
# Append extension data
if 'predicted_test' in locals():
    train = train.append([predicted_test], sort=False)

if 'extension' in locals():
    train = train.append([extension], sort=False)

# Conbine train and test for preprocessing
df = train.drop(columns=target).append(test, sort=False)

# %%
# Imputation
# rw dtype info to avoid converting problem of Orange3
imputation_strategy = refn.imputation_strategy(classified_features)
df = refn.impute_features(df, imputation_strategy)

# Restore dtype
df[classified_features['int']] = df[classified_features['int']].astype('int64')
df[classified_features['float']] = df[classified_features['float']].astype('float64')

for r in imputation_strategy:
    print(r)
    print('\n')

del(imputation_strategy)

# %%
# Add
# Additional features
addition = refn.additional_featues(df)
addition = pd.concat([addition, refn.up_low_in_neighborhood(df)], axis=1)

df = pd.concat([df, addition], axis=1)

# %%
# Dimensionality reduction
# Replace and combine levels in each feature.
# Before encoding the category values, I replaced them with words for analysis.
dr_strategy = refn.dimensionality_reduction_strategy()
refn.reduce_dimensions(df, dr_strategy)

for r in dr_strategy:
    print(r)
    print(dr_strategy[r])
    print('\n')

del(dr_strategy)

# %%
# Encoding
ordinal_features = refn.ordinaly_encoding_strategy(df)
refn.encode_categories(df, ordinal_features)

print(ordinal_features)

del(ordinal_features)

# %%
# Scaling
# Keep features for OneHot encoding
onehot_strategy = refn.onehot_encoding_strategy(df)
onehot_features = df[['District', *onehot_strategy]]

# Standardization, or mean removal and variance scaling
sc_strategy = refn.scaling_strategy(df, classified_features)
df = refn.scaled_features(df, sc_strategy)

for r in sc_strategy:
    print(r)
    print(sc_strategy[r])
    print('\n')

del(sc_strategy)

# %%
# OneHot
df = pd.concat([df, pd.get_dummies(onehot_features)], axis=1)

del(onehot_features)

# %%
# Drop
# Features selection
# Remove columns which are not feature, and features with low variance.
drop_columns = refn.drop_strategy(df)
final_df = df.drop(columns=drop_columns, errors='ignore')

del(drop_columns)

# %%
# Garbage collection
import gc
gc.collect()

# %%
# Set X
X = final_df.iloc[:len(train), :]

test = final_df.iloc[len(train):, :]
test = test.drop(columns=target, errors='ignore')

# %%
# Set y
y = train[target]
y = np.log1p(y)

# %%
# create estimator
activation = ['identity', 'logistic', 'tanh', 'relu'] # default='relu'
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'] # default='auto'
alpha_rvs = sp.stats.expon.rvs(scale=1)
average = [True, False] # default=False
batch_size = ['auto']
bootstrap = [True, False] # default=False
bootstrap_features = [True, False] # default=False
C_rvs = sp.stats.expon(scale=1)
class_weight = [None, 'dict', 'balanced'] # default=None
criterion = ['mae', 'mse', 'friedman_mse'] # default='mse'
compute_score = [True, False] # default=True
dual = [True, False] # default=True, LogisticRegression's default=False
early_stopping = [True, False] # default=False
fit_intercept = [True, False] # default=True
gamma = ['auto', 'scale']
gamma_rvs = sp.stats.expon.rvs(scale=0.01)
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
l1_ratio_rvs = sp.stats.expon.rvs(scale=1) # default=None
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive'] # default='invscaling'
loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
loss_for_ada = ['linear', 'square', 'exponential']
loss_for_ransac = ['absolute_loss', 'squared_loss']
max_features = ['auto', 'sqrt', 'log2'] # default=None. 'auto' is the same as None.
max_iter_rvs = sp.stats.randint(100, 3000)
metric = ['minkowski']
metric_params = [None]
max_depth_rvs = sp.stats.randint(1, 101)
max_features = [None, 'auto', 'sqrt', 'log2']
min_samples_split_rvs = sp.stats.randint.rvs(2, 11)
min_samples_leaf_rvs = sp.stats.randint.rvs(1, 11)
multi_class = ['ovr', 'multinomial', 'auto'] # default='ovr'
nesterovs_momentum = [True, False]
normalize = [True, False] # default=False
normalize_y = [True, False] # default=False
oob_score = [True, False] # default=False
optimizer = ['fmin_l_bfgs_b']
penalty = ['l1', 'l2', 'elasticnet', 'none'] # default='l2'
positive = [True, False] # default=False
precompute = [True, False] # default=False
presort = [True, False] # default=False
selection = ['cyclic', 'random'] # default='cyclic'
solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'] # default='auto'
splitter = ['best', 'random'] # default='best'
shuffle = [True, False] # default=True
shrinking = [True, False]
tol_rvs = sp.stats.expon.rvs(scale=0.0001)
random_state = [None, 42] # default=None
warm_start = [True, False] # default=False
weights = ['uniform', 'distance'] #
# XGM
booster = ['gbtree', 'gblinear', 'dart']
# LGBM
boosting_type = ['gbdt', 'dart', 'goss', 'rf']
objective = [None, 'regression', 'binary', 'lambdarank']

# %%
%%time
from sklearn.tree import DecisionTreeRegressor

estimator = DecisionTreeRegressor()
if search_hyperparameter:
    parameters = {}
else:
    parameters = {
        'criterion': ['friedman_mse'],
        'max_depth': [10],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0.003, 0.004, 0.005, 0.006, 0.007],
        'min_impurity_split': [1e-7, 4e-7, 5e-7, 6e-7, 1e-6, 5e-6],
        'min_samples_leaf': [5, 6, 7, 8, 9],
        'min_samples_split': [3],
        'min_weight_fraction_leaf': [0, 1e-12, 1e-11, 1e-10, 5e-10, 1e-9],
        'presort': [True],
        'splitter': ['random']
        # 'random_state': random_state,
    }

gscv_dtr = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-12 01:10:52
best score=0.76996003353969
best params={'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 1e-07, 'min_samples_leaf': 2, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'best'}
2. 2019-10-12 01:52:19
best score=0.772390823153533
best params={'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 1e-06, 'min_samples_leaf': 4, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
Wall time: 21.7 s
3. 2019-10-12 10:17:49
best score=0.7800101983334903
best params={'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 1e-08, 'min_samples_leaf': 6, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
4. 2019-10-12 10:26:09
best score=0.7945675710804673
best params={'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0001, 'min_impurity_split': 1e-07, 'min_samples_leaf': 7, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
5. 2019-10-12 10:57:10
best score=0.796532493447863
best params={'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.01, 'min_impurity_split': 1e-06, 'min_samples_leaf': 5, 'min_samples_split': 3, 'min_weight_fraction_leaf': 1e-06, 'presort': True, 'random_state': None, 'splitter': 'random'}
Wall time: 18min 58s
6. 2019-10-13 14:05:54
Model with rank: 1
Mean validation score: 0.798 (std: 0.009)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.001, 'min_impurity_split': 1e-05, 'min_samples_leaf': 5, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 2
Mean validation score: 0.796 (std: 0.005)
Parameters: {'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 4.9999999999999996e-06, 'min_samples_leaf': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 1e-07, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 3
Mean validation score: 0.793 (std: 0.017)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 5e-07, 'min_samples_leaf': 8, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
7. 2019-10-13 05:50:32
Model with rank: 1
Mean validation score: 0.799 (std: 0.013)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.005, 'min_impurity_split': 5e-07, 'min_samples_leaf': 8, 'min_samples_split': 2, 'min_weight_fraction_leaf': 1e-07, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 2
Mean validation score: 0.797 (std: 0.020)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0005, 'min_impurity_split': 1e-06, 'min_samples_leaf': 8, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 3
Mean validation score: 0.797 (std: 0.012)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.05, 'min_impurity_split': 1e-07, 'min_samples_leaf': 8, 'min_samples_split': 4, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
8. 2019-10-14 02:00:37
Model with rank: 1
Mean validation score: 0.802306 (std: 0.012524)
Parameters: {'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0, 'min_impurity_split': 1e-05, 'min_samples_leaf': 6, 'min_samples_split': 3, 'min_weight_fraction_leaf': 1e-09, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 2
Mean validation score: 0.797761 (std: 0.005755)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.001, 'min_impurity_split': 5e-06, 'min_samples_leaf': 5, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'presort': True, 'random_state': None, 'splitter': 'random'}
Model with rank: 3
Mean validation score: 0.797727 (std: 0.011658)
Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0005, 'min_impurity_split': 1e-05, 'min_samples_leaf': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 1e-07, 'presort': True, 'random_state': None, 'splitter': 'random'}
"""

# %%
submit_prediction(gscv_dtr, test)

# %%
%%time
from sklearn.linear_model import BayesianRidge

estimator = BayesianRidge()
parameters = {
    #'n_iter': [1, 300, 1000],
    #'alpha_1': [1e-6],
    #'alpha_2': [1e-6],
    #'lambda_1': [1e-6],
    #'lambda_2': [1e-6],
    #'compute_score': compute_score,
    #'fit_intercept': fit_intercept,
    #'normalize': normalize
}

gscv_bysr = grid_search(estimator, parameters, X, y)

''' Results
1. bayes_2019-10-16-05-18-21.gscv
Model with rank: 1
Mean validation score: 0.901667 (std: 0.007357)
Parameters: {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'compute_score': all, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'n_iter': all, 'normalize': False}
2. bayes_2019-10-16-05-22-34.gscv
Model with rank: 1
Mean validation score: 0.901667 (std: 0.007357)
Parameters: {'alpha_1': 1e-05, 'alpha_2': 1e-07, 'compute_score': True, 'lambda_1': 1e-07, 'lambda_2': 1e-05, 'n_iter': all, 'normalize': False}
3. bayes_2019-10-16-05-33-36.gscv
Model with rank: 1
Mean validation score: 0.901667 (std: 0.007357)
Parameters: {}
Wall time: 355 ms
'''


# %%
%%time
from sklearn.linear_model import ElasticNet

estimator = ElasticNet()
parameters_random = {
    'alpha': alpha_rvs,
    'l1_ratio': l1_ratio_rvs
}

rscv_elsn = random_search(estimator, parameters_random, X, y, 5)

# %%
%%time
from sklearn.linear_model import ElasticNet

estimator = ElasticNet()
parameters_random = {
    'alpha': sp.stats.expon(scale=1),
    'l1_ratio': sp.stats.expon(scale=1),
    'fit_intercept': fit_intercept,
    'normalize': normalize,
    'precompute': precompute,
    'max_iter': sp.stats.randint(100, 3000),
    'tol': sp.stats.expon(scale=1),
    'warm_start': warm_start,
    'positive': positive,
    'selection': selection,
    'random_state': random_state,
}

rscv_elsn = random_search(estimator, parameters_random, X, y, 5)

# %%
parameters_grid = {
    'alpha': [0.0009, 0.0010, 0.0011],
    'l1_ratio': [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.050, 0.100, 0.150],
    # 'fit_intercept': fit_intercept,
    # 'normalize': normalize,
    'precompute': [True],
    'max_iter': [725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125],
    'tol': [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085],
    # 'warm_start': warm_start,
    'positive': [True],
    'selection': ['random'],
    # 'random_state': random_state,
}

gscv_elsn = grid_search(estimator, parameters_grid, X, y)


""" Results
1. 2019-10-12 02:37:10
best score=0.8229924859073431
best params={'alpha': 0.5, 'fit_intercept': True, 'l1_ratio': 0.0, 'max_iter': 500, 'normalize': False, 'positive': False, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.0001, 'warm_start': False}
2. 2019-10-12 02:54:08
best score=0.90306339433955
best params={'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.0, 'max_iter': 600, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.0001, 'warm_start': False}
3. 2019-10-12 03:03:27
best score=0.9030818941101596
best params={'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.01, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.001, 'warm_start': False}
4. 2019-10-12 03:09:44
best score=0.9030960342337475
best params={'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.01, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.01, 'warm_start': False}
Wall time: 2min 6s
5. 2019-10-12 11:13:51
best score=0.9031927620771343
best params={'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.1, 'max_iter': 900, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.01, 'warm_start': False}
Wall time: 3min 2s
6. 2019-10-12 11:23:59
best score=0.9031687235593935
best params={'alpha': 0.0001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 700, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.01, 'warm_start': False}
Wall time: 4min 16s
7. 2019-10-13 06:15:24
Model with rank: 1
Mean validation score: 0.904 (std: 0.005)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.005, 'max_iter': 900, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.904 (std: 0.005)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'l1_ratio': 0.0, 'max_iter': 1100, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.1, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.903 (std: 0.006)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.1, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
8. 2019-10-14 02:05:31
Model with rank: 1
Mean validation score: 0.903864 (std: 0.005193)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'l1_ratio': 0.5, 'max_iter': 700, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.903768 (std: 0.005657)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.01, 'max_iter': 700, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.903723 (std: 0.005852)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.001, 'max_iter': 1100, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Wall time: 1min 30s
9. 2019-10-14 11:49:58
Model with rank: 1
Mean validation score: 0.904330 (std: 0.005900)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.01, 'max_iter': 850, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.903912 (std: 0.004743)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.0, 'max_iter': 1050, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.06, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.903887 (std: 0.004777)
Parameters: {'alpha': 0.001, 'fit_intercept': True, 'l1_ratio': 0.001, 'max_iter': 750, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.08, 'warm_start': False}
Wall time: 16min 39s (On charlotte)
coordinate_descent_2019-10-17-08-29-07.gscv
Model with rank: 1
Mean validation score: 0.904835 (std: 0.006049)
Parameters: {'alpha': 0.001, 'l1_ratio': 0.005, 'max_iter': 850, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.08, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.904204 (std: 0.004363)
Parameters: {'alpha': 0.001, 'l1_ratio': 0.01, 'max_iter': 1050, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.08, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.904111 (std: 0.005494)
Parameters: {'alpha': 0.001, 'l1_ratio': 0.05, 'max_iter': 850, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.06, 'warm_start': False}
"""


# %%
%%time
from sklearn.linear_model import ElasticNetCV

elnscv = ElasticNetCV()
parameters = {
    'l1_ratio': [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.050, 0.100, 0.150],
    # 'eps' = [0.001],
    # 'n_alphas' = [100],
    # 'alphas': [0.0009, 0.0010, 0.0011],
    'fit_intercept': fit_intercept,
    'normalize': normalize,
    'precompute': ['auto'],
    'max_iter': [725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125],
    'tol': [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085],
    'cv': CV_,
    'verbose': VERBOSE_,
    'n_jobs': N_JOBS_,
    'positive': positive,
    'selection': selection,
    'random_state': random_state,
}

elnscv.set_params(parameters)
elnscv.fit(X, y)


# %%
# Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
%%time
from sklearn.linear_model import MultiTaskElasticNet

estimator = MultiTaskElasticNet()
parameters = {
    'alpha': [0.0009, 0.0010, 0.0011],
    'l1_ratio': [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.050, 0.100, 0.150],
    'fit_intercept': fit_intercept,
    'normalize': normalize,
    'max_iter': [725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125],
    'tol': [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085],
    'warm_start': warm_start,
    'random_state': random_state,
    'selection': selection,
}

gscv_mlt_eln = grid_search(estimator, parameters, X, y)


""" Results
1. ValueError: For mono-task outputs, use ElasticNet
"""

# %%
%%time
from sklearn.linear_model import Lasso

estimator = Lasso()
parameters_random = {
    'alpha': sp.stats.expon(scale=1),
    'fit_intercept': fit_intercept,
    'normalize': normalize,
    'precompute': precompute,
    'max_iter': sp.stats.randint(100, 3000),
    'tol': sp.stats.expon(scale=1),
    'warm_start': warm_start,
    'positive': positive,
    'selection': selection,
    'random_state': random_state,
}

rscv_lsso = random_search(estimator, parameters_random, X, y, 5)

# %%
estimator = Lasso()
parameters = {
    'alpha': [5e-6, 1e-5, 6e-5, 5e-5, 4e-5, 2e-5, 1e-4, 8e-4, 6e-5, 5e-4, 1e-3],
    # 'fit_intercept': fit_intercept,
    'normalize': normalize,
    'precompute': [True],
    'max_iter': [800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050],
    'tol': [1e-3, 2e-3, 1e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-1, 9e-1, 8e-1, 6e-1, 5e-1],
    'warm_start': warm_start,
    'positive': positive,
    # 'random_state': random_state,
    'selection': ['random'],
}

gscv_lsso = grid_search(estimator, parameters, X, y)


""" Results
1. 2019-10-12 03:48:40
best score=0.903365760240962
best params={'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 1000, 'normalize': True, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.01, 'warm_start': False}
2. 2019-10-12 03:56:00
best score=0.9033262127451739
best params={'alpha': 1e-05, 'fit_intercept': True, 'max_iter': 900, 'normalize': True, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.01, 'warm_start': False}
Wall time: 1min 40s
3. 2019-10-12 11:29:15
best score=0.9033470764941293
best params={'alpha': 1e-05, 'fit_intercept': True, 'max_iter': 1000, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.1, 'warm_start': False}
Wall time: 46.6 s
4. 2019-10-12 11:31:53
best score=0.9038025673472447
best params={'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Wall time: 42.7 s
5. 2019-10-13 07:25:49
Model with rank: 1
Mean validation score: 0.904 (std: 0.006)
Parameters: {'alpha': 1e-05, 'fit_intercept': True, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.904 (std: 0.006)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.904 (std: 0.005)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 900, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.1, 'warm_start': False}
6. 2019-10-14 02:09:01
Model with rank: 1
Mean validation score: 0.903371 (std: 0.008326)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 1000, 'normalize': True, 'positive': False, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.05, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.903352 (std: 0.005946)
Parameters: {'alpha': 5e-05, 'fit_intercept': True, 'max_iter': 800, 'normalize': True, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.001, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.903340 (std: 0.005899)
Parameters: {'alpha': 5e-05, 'fit_intercept': True, 'max_iter': 900, 'normalize': True, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.005, 'warm_start': False}
Wall time: 14.5 s
7. 2019-10-14 11:58:11
Model with rank: 1
Mean validation score: 0.904075 (std: 0.005349)
Parameters: {'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 800, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.04, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.903890 (std: 0.005861)
Parameters: {'alpha': 1e-05, 'fit_intercept': True, 'max_iter': 1050, 'normalize': True, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.04, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.903821 (std: 0.004850)
Parameters: {'alpha': 5e-06, 'fit_intercept': True, 'max_iter': 950, 'normalize': False, 'positive': True, 'precompute': True, 'random_state': None, 'selection': 'random', 'tol': 0.06, 'warm_start': False}
Wall time: 1min 5s
8. coordinate_descent_2019-10-17-08-43-40.gscv
Model with rank: 1
Mean validation score: 0.904538 (std: 0.005698)
Parameters: {'alpha': 5e-05, 'max_iter': 1000, 'normalize': False, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.07, 'warm_start': True}
Model with rank: 2
Mean validation score: 0.904385 (std: 0.003944)
Parameters: {'alpha': 5e-06, 'max_iter': 925, 'normalize': True, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.04, 'warm_start': True}
Model with rank: 3
Mean validation score: 0.903953 (std: 0.005299)
Parameters: {'alpha': 6e-05, 'max_iter': 925, 'normalize': False, 'positive': True, 'precompute': True, 'selection': 'random', 'tol': 0.04, 'warm_start': False}
"""

# %%
%%time
from sklearn.linear_model import Ridge

estimator = Ridge()
parameters = {
    'alpha': [1,49, 1.5, 1,51],
    # 'fit_intercept': fit_intercept,
    # 'normalize': normalize,
    # 'max_iter': [100, 200, 300],
    'tol': [1e-3, 4e-3, 2e-3, 1e-2, 9e-2, 8e-2, 6e-2, 1e-1, 1],
    'solver': solver,
    # 'random_state': random_state,
}

gscv_rgd = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-12 04:52:28
best score=0.9008655207439684
best params={'alpha': 0.01, 'fit_intercept': True, 'max_iter': 700, 'normalize': False, 'random_state': None, 'solver': 'lsqr', 'tol': 0.01}
2. 2019-10-12 05:04:16
best score=0.9008944484741844
best params={'alpha': 0.1, 'fit_intercept': True, 'max_iter': 300, 'normalize': False, 'random_state': None, 'solver': 'lsqr', 'tol': 0.01}
3. 2019-10-12 05:06:07
best score=0.9008944484741844
best params={'alpha': 0.1, 'fit_intercept': True, 'max_iter': 100, 'normalize': False, 'random_state': None, 'solver': 'lsqr', 'tol': 0.01}
Wall time: 35.3 s
4. 2019-10-12 11:38:32
best score=0.901649278149774
best params={'alpha': 1, 'fit_intercept': True, 'max_iter': 50, 'normalize': False, 'random_state': None, 'solver': 'sparse_cg', 'tol': 0}
Wall time: 2min 52s
5. 2019-10-12 11:44:17
best score=0.9017143291811528
best params={'alpha': 2, 'fit_intercept': True, 'max_iter': 50, 'normalize': False, 'random_state': None, 'solver': 'lsqr', 'tol': 0.001}
Wall time: 1min 5s
6. 2019-10-12 11:51:54
best score=0.9017143291811528
best params={'alpha': 2, 'fit_intercept': True, 'max_iter': 40, 'normalize': False, 'random_state': None, 'solver': 'lsqr', 'tol': 0.001}
Wall time: 1min 10s
7. 2019-10-13 07:46:02
Model with rank: 1
Mean validation score: 0.901758 (std: 0.007555)
Parameters: {'alpha': 1.5, 'fit_intercept': True, 'max_iter': all, 'normalize': False, 'random_state': None, 'solver': 'sparse_cg', 'tol': 0.001}
8. 2019-10-14 02:16:26
Model with rank: 1
Mean validation score: 0.901758 (std: 0.007555)
Parameters: {'alpha': 1.5, 'fit_intercept': True, 'max_iter': all, 'normalize': False, 'random_state': None, 'solver': 'sparse_cg', 'tol': 0.001}
Wall time: 1min 53s
9. 2019-10-14 12:08:08
Model with rank: 1
Mean validation score: 0.901759 (std: 0.007554)
Parameters: {'alpha': 1.5, 'fit_intercept': True, 'max_iter': all, 'normalize': False, 'random_state': None, 'solver': 'sparse_cg', 'tol': 0.001}
Wall time: 1min 22s (On charlotte)
"""


# %%
%%time
from sklearn.linear_model import LinearRegression

estimator = LinearRegression()
parameters = {
    'fit_intercept': [True, False],
    'normalize': normalize
}

gscv_lir = grid_search(estimator, parameters, X, y)

""" Results
1.
best params={'fit_intercept': True, 'normalize': False}
2.
best score=0.8998849997882236
best params={'fit_intercept': True, 'normalize': True}
Wall time: 2.75 s
3. 2019-10-14 13:16:36
Model with rank: 1
Mean validation score: 0.899885 (std: 0.007852)
Parameters: {'fit_intercept': True, 'normalize': True}
Model with rank: 1
Mean validation score: 0.899885 (std: 0.007852)
Parameters: {'fit_intercept': True, 'normalize': False}
Model with rank: 3
Mean validation score: -0.008698 (std: 0.155931)
Parameters: {'fit_intercept': False, 'normalize': True}
Model with rank: 3
Mean validation score: -0.008698 (std: 0.155931)
Parameters: {'fit_intercept': False, 'normalize': False}
Wall time: 1.53 s (On charlotte)
"""


# %%
%%time
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
parameters = {
    'penalty': penalty,
    'dual': dual,
    'tol': [1e-3, 1e-4, 1e-5],
    'C': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    # 'fit_intercept': fit_intercept,
    'intercept_scaling': [0.9, 1.0, 1.1],
    'class_weight': class_weight,
    # 'random_state': random_state,
    'solver': solver,
    'max_iter': [50, 100, 150],
    'multi_class': multi_class,
    'warm_start': [False],
    'l1_ratio': l1_ratio
}

gscv_lgr = grid_search(estimator, parameters, X, y)

''' Results
1.
best score=0.01577503429355281
best params={'C': 0.7, 'dual': True, 'max_iter': 10}
2. 2019-10-11
ValueError: Unknown label type: 'continuous'
3.
ValueError: Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
'''


# %%
# The Huber Regressor is Linear regression model that is robust to outliers.
%%time
from sklearn.linear_model import HuberRegressor

estimator = HuberRegressor()
parameters = {
    'epsilon': [2.2, 2.25, 2.27, 2.3, 2.32, 2.35, 2.4, 2.45],
    'max_iter': [280, 285, 290, 295, 300, 310],
    'alpha': [5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 9e-3, 8e-3, 1e-2],
    # 'warm_start': [True, False(default)],
    # 'fit_intercept': [True(default), False],
    # 'tol': [1e-5(default), 1e-1],
}

gscv_hbr = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-12 05:54:45
best score=0.9011997587784956
best params={'alpha': 0.0001, 'epsilon': 2.4, 'fit_intercept': True, 'max_iter': 300, 'tol': 1, 'warm_start': False}
Wall time: 36min 47s
2. 2019-10-13 08:29:43
Model with rank: 1
Mean validation score: 0.901200 (std: 0.005978)
Parameters: {'alpha': 0.0001, 'epsilon': 2.4, 'fit_intercept': True, 'max_iter': 300, 'tol': all, 'warm_start': False}
Wall time: 37min 49s
3. 2019-10-14 00:30:11
Model with rank: 1
Mean validation score: 0.901203 (std: 0.006288)
Parameters: {'alpha': 0.01, 'epsilon': 2.2, 'fit_intercept': True, 'max_iter': 300, 'tol': all, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.901200 (std: 0.005978)
Parameters: {'alpha': 0.0001, 'epsilon': 2.4, 'fit_intercept': True, 'max_iter': 300, 'tol': all, 'warm_start': False}
Wall time: 2min 29s
4. 2019-10-14 12:51:59
Model with rank: 1
Mean validation score: 0.900880 (std: 0.006442)
Parameters: {'alpha': 0.0005, 'epsilon': 2.35, 'fit_intercept': True, 'max_iter': 300, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.900805 (std: 0.006406)
Parameters: {'alpha': 0.0005, 'epsilon': 2.25, 'fit_intercept': True, 'max_iter': 325, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.900792 (std: 0.006055)
Parameters: {'alpha': 0.005, 'epsilon': 2.35, 'fit_intercept': True, 'max_iter': 300, 'warm_start': False}
Wall time: 3min 14s (On charlotte)
5. 2019-10-14 13:07:31
Model with rank: 1
Mean validation score: 0.901301 (std: 0.004902)
Parameters: {'alpha': 0.0008, 'epsilon': 2.4, 'max_iter': 280}
Model with rank: 2
Mean validation score: 0.901192 (std: 0.006480)
Parameters: {'alpha': 0.0003, 'epsilon': 2.4, 'max_iter': 300}
Model with rank: 3
Mean validation score: 0.901143 (std: 0.006811)
Parameters: {'alpha': 0.01, 'epsilon': 2.25, 'max_iter': 280}
Wall time: 6min 1s (On charlotte)
"""


# %%
%%time
from sklearn.linear_model import PassiveAggressiveRegressor

estimator = PassiveAggressiveRegressor()
parameters = {
    'C': [0.6, 0.65, 0.7, 0.8],
    # 'fit_intercept': [True(default), False],
    'max_iter': [300, 400, 450, 500, 550],
    'tol': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    'early_stopping': [False],
    'validation_fraction': [1e-3, 1e-2, 1e-1, 5e-1],
    'n_iter_no_change': [5, 6, 7],
    'shuffle': shuffle,
    'loss': loss,
    'epsilon': [1e-2, 5e-2, 1e-1, 5e-1,],
    # 'random_state': random_state,
    'warm_start': warm_start,
    'average': average
}

gscv_pasr = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-12 09:35:31
best score=0.8799520404489535
best params={'C': 0.8, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 600, 'n_iter_no_change': 6, 'random_state': None, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
Wall time: 28min 50s
2. 2019-10-12 10:09:54
best score=0.88453217144599
best params={'C': 0.7, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 400, 'n_iter_no_change': 6, 'random_state': None, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
Wall time: 23min 34s
3. 2019-10-13 09:26:43
Model with rank: 1
Mean validation score: 0.878117 (std: 0.004594)
Parameters: {'C': 0.6, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 200, 'n_iter_no_change': 6, 'random_state': None, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'warm_start': False}
Model with rank: 2
Mean validation score: 0.877292 (std: 0.011384)
Parameters: {'C': 0.7, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 300, 'n_iter_no_change': 6, 'random_state': None, 'shuffle': True, 'tol': 1e-06, 'validation_fraction': 0.1, 'warm_start': False}
Model with rank: 3
Mean validation score: 0.875541 (std: 0.006902)
Parameters: {'C': 0.7, 'average': False, 'early_stopping': False, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 500, 'n_iter_no_change': 7, 'random_state': None, 'shuffle': True, 'tol': 0.0001, 'validation_fraction': 0.01, 'warm_start': False}
Wall time: 42min 46s
"""


# %%
%%time
from sklearn.linear_model import SGDRegressor

estimator = SGDRegressor()
parameters = {
    'loss': loss,
    'penalty': penalty,
    'alpha': [1e-5, 1e-4],
    'l1_ratio': [0, 0.15, 0.5, 0.85, 1],
    # 'fit_intercept': fit_intercept,
    'max_iter': [1000],
    'tol': [1e-4, 1e-3, 1e-2],
    'shuffle': shuffle,
    'epsilon': [1e-3, 1e-2, 1e-1],
    # 'random_state': random_state,
    'learning_rate': learning_rate,
    'eta0': [1e-3, 1e-2, 1e-1],
    'power_t': [0.1, 0.5, 1],
    'validation_fraction': [0.1, 0.5, 0.9],
    'n_iter_no_change': [1, 5, 10],
    'warm_start': warm_start,
    'average': average,
}

gscv_sdg = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-12 21:39:55
best score=0.6856429154288098
best params={'alpha': 0.0001, 'average': True, 'epsilon': 0.1, 'eta0': 0.1, 'fit_intercept': True, 'l1_ratio': 1, 'learning_rate': 'constant', 'loss': 'epsilon_insensitive', 'max_iter': 1000, 'n_iter_no_change': 10, 'penalty': 'none', 'power_t': 0.1, 'random_state': None, 'shuffle': True, 'tol': 0.0001, 'validation_fraction': 0.9, 'warm_start': False}
Wall time: 51min 50s
"""


# %%
%%time
from sklearn.linear_model import TheilSenRegressor

estimator = TheilSenRegressor()
parameters = {
    'fit_intercept': fit_intercept,
    'max_subpopulation': [1e3, 1e4, 1e5],
    'n_subsamples': [None, X.shape[1], int(X.shape[0]/2), X.shape[0]],
    'max_iter': [200, 300, 400, 500],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1],
    # 'random_state': random_state,
}

gscv_tlsn = grid_search(estimator, parameters, X, y)

""" Results
1. "timeout or by a memory leak.", UserWarning
2. "timeout or by a memory leak.", UserWarning
3. "timeout or by a memory leak.", UserWarning
"""

# %%
%%time
from sklearn.svm import SVR

estimator = SVR()
parameters = {
    'kernel': kernel,
    'degree': [1, 3, 5, 7, 9],
    'gamma': gamma,
    'coef0': [0, 1e-3, 1e-2, 1e-1, 1],
    'tol': [0, 1e-4, 1e-3, 1e-2],
    'C': [5e-1, 1.0, 1.5],
    'epsilon': [1e-3, 1e-2, 1e-1],
    'shrinking': shrinking,
    'max_iter': [-1],
}

gscv_svr = grid_search(estimator, parameters, X, y)

""" Results
1. ValueError: X should be a square kernel matrix
"""

# %%
%%time
from sklearn.svm import LinearSVR

estimator = LinearSVR()
parameters = {
    'epsilon': [0.09, 0.1, 0.11, 0.12],
    'C': [1.39, 1.4, 1.41],
    #'loss': ['epsilon_insensitive'(default), 'squared_epsilon_insensitive'],
    #'fit_intercept': [True, False],
    'intercept_scaling': [3.1],
    #'dual': [True(default), False],
    #'random_state': [None(default)],
    'max_iter': [2200],
    'tol': [0.00009, 0.0001, 0.0002, 0.0003]
}

gscv_lsvr = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-13 11:32:40
Model with rank: 1
Mean validation score: 0.897535 (std: 0.004983)
Parameters: {'C': 1.5, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 1.5, 'loss': 'epsilon_insensitive', 'max_iter': 1500, 'random_state': None, 'tol': 1e-05}
Model with rank: 2
Mean validation score: 0.897417 (std: 0.006957)
Parameters: {'C': 1.5, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 1.5, 'loss': 'epsilon_insensitive', 'max_iter': 1000, 'random_state': None, 'tol': 1e-05}
Model with rank: 3
Mean validation score: 0.897095 (std: 0.005929)
Parameters: {'C': 1.5, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 1.5, 'loss': 'epsilon_insensitive', 'max_iter': 1500, 'random_state': None, 'tol': 0.0001}
Wall time: 6min 29s
2. 2019-10-14 00:21:17
Model with rank: 1
Mean validation score: 0.899967 (std: 0.006747)
Parameters: {'C': 1.5, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 2, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 5e-05}
Model with rank: 2
Mean validation score: 0.899911 (std: 0.006732)
Parameters: {'C': 2, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 1.5, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 1e-06}
Model with rank: 3
Mean validation score: 0.899634 (std: 0.005227)
Parameters: {'C': 1.0, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 2, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 1e-05}
Wall time: 4min 31s
3. 2019-10-14 03:35:44
Model with rank: 1
Mean validation score: 0.900916 (std: 0.007511)
Parameters: {'C': 1, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 0.0001}
Model with rank: 2
Mean validation score: 0.900779 (std: 0.007552)
Parameters: {'C': 1, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 5e-05}
Model with rank: 3
Mean validation score: 0.900717 (std: 0.007436)
Parameters: {'C': 1, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3, 'loss': 'epsilon_insensitive', 'max_iter': 2000, 'random_state': None, 'tol': 0.001}
Wall time: 1h 14min 21s
4. 2019-10-14 05:31:48
Model with rank: 1
Mean validation score: 0.901147 (std: 0.008760)
Parameters: {'C': 1.4, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3.1, 'loss': 'epsilon_insensitive', 'max_iter': 2200, 'random_state': None, 'tol': 1e-05}
Model with rank: 2
Mean validation score: 0.901135 (std: 0.008788)
Parameters: {'C': 1.2, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3.2, 'loss': 'epsilon_insensitive', 'max_iter': 1800, 'random_state': None, 'tol': 0.0009}
Model with rank: 3
Mean validation score: 0.901131 (std: 0.008291)
Parameters: {'C': 0.9, 'dual': True, 'epsilon': 0.1, 'fit_intercept': True, 'intercept_scaling': 3.3, 'loss': 'epsilon_insensitive', 'max_iter': 1600, 'random_state': None, 'tol': 4e-05}
Wall time: 1h 46min 12s
5. classes_2019-10-14-19-26-00.gscv
Model with rank: 1
Mean validation score: 0.902062 (std: 0.008405)
Parameters: {'C': 1.39, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 0.0001}
Model with rank: 2
Mean validation score: 0.901461 (std: 0.008384)
Parameters: {'C': 1.4, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 0.0002}
Model with rank: 3
Mean validation score: 0.901319 (std: 0.008010)
Parameters: {'C': 1.4, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 0.0001}
Wall time: 46.9 s
6. classes_2019-10-17-10-01-34.gscv
Model with rank: 1
Mean validation score: 0.901689 (std: 0.008863)
Parameters: {'C': 1.39, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 9e-05}
Model with rank: 2
Mean validation score: 0.901507 (std: 0.006854)
Parameters: {'C': 1.4, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 0.0002}
Model with rank: 3
Mean validation score: 0.901478 (std: 0.007016)
Parameters: {'C': 1.4, 'epsilon': 0.09, 'intercept_scaling': 3.1, 'max_iter': 2200, 'tol': 9e-05}
"""


# %%
%%time
from sklearn.svm import NuSVR

estimator = NuSVR()
parameters = {
    'nu': [1e-1, 0.5, 1],
    'C': [5e-1, 1.0, 1.5],
    'kernel': kernel,
    'degree': [1, 3, 5, 7, 9],
    'gamma': ,
    'coef0': [0, 1e-3, 1e-2, 1e-1, 1],
    'shrinking': shrinking,
    'tol': [0, 1e-4, 1e-3, 1e-2],
    'max_iter': [-1]
}

gscv_nsvr = grid_search(estimator, parameters, X, y)

""" Results
1. ValueError: X should be a square kernel matrix
"""


# %%
# KNeighborsRegressor is Regression based on k-nearest neighbors.
%%time
from sklearn.neighbors import KNeighborsRegressor

estimator = KNeighborsRegressor(n_neighbors=2)
parameters = {
    'n_neighbors': [10],
    'weights': weights,
    'algorithm': algorithm,
    'leaf_size': [1, 5, 10, 15, 30],
    'p': [1, 2],
    'metric': metric,
    'metric_params': metric_params
}

gscv_kneib = grid_search(estimator, parameters, X, y)

""" Results
1.
best score=0.8289400733086824
best params={'algorithm': 'brute', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 8, 'p': 1, 'weights': 'distance'}
2.
best score=0.8297282806747166
best params={'algorithm': 'auto', 'leaf_size': 1, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 14, 'p': 1, 'weights': 'distance'}
3.
2019-10-13 10:42:37
Model with rank: 1
Mean validation score: 0.829728 (std: 0.008315)
Parameters: {'algorithm': all, 'leaf_size': all, 'metric': 'minkowski', 'metric_params': None, 'n_neighbors': 14, 'p': 1, 'weights': 'distance'}
Wall time: 54min 28s

Notes :
If n_neighbors is 14, it seems overfitting. 
"""



# %%
%%time
from sklearn.neighbors import RadiusNeighborsRegressor

estimator = RadiusNeighborsRegressor()
parameters = {
    'radius': [0.5, 1.0, 1.5],
    'weights': weights,
    'algorithm': algorithm,
    'leaf_size': [10, 30, 50],
    'p': [1, 2, 3],
    'metric': metric,
    'metric_params': metric_params
}

gscv_rneib = grid_search(estimator, parameters, X, y)

''' Results
1.
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
'''

# %%
%%time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, RBF, WhiteKernel

kernels = [
    DotProduct(sigma_0=1),
    # WhiteKernel(noise_level=1),
    DotProduct(sigma_0=1) + WhiteKernel(noise_level=1),
    # 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
    # 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
    # 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)),
    # ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
    # ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
    # 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
]

estimator = GaussianProcessRegressor()
parameters = {
    'kernel': kernels,
    'alpha': [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1],
    # 'optimizer': optimizer,
    'n_restarts_optimizer': [1, 2, 3, 4, 5],    # 'n_restarts_optimizer': [0(default), 1, 2, 3, 4],
    # 'normalize_y': normalize_y,
    # 'random_state': random_state
}

gscv_gusp = grid_search(estimator, parameters, X, y)

""" Results
1.
best score=0.9018426177687819
best params={'alpha': 1.0, 'copy_X_train': True, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 1, 'normalize_y': False, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None}
2. 2019-10-14 01:40:37
Model with rank: 1
Mean validation score: 0.901843 (std: 0.007085)
Parameters: {'alpha': 1.0, 'copy_X_train': True, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 2, 'normalize_y': False, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None}
Model with rank: 2
Mean validation score: 0.901843 (std: 0.007085)
Parameters: {'alpha': 1.0, 'copy_X_train': True, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 1, 'normalize_y': False, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None}
Model with rank: 3
Mean validation score: 0.901843 (std: 0.007085)
Parameters: {'alpha': 1.0, 'copy_X_train': True, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 0, 'normalize_y': False, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None}
Wall time: 1h 4min 53s
3. 2019-10-14 13:50:42
Model with rank: 1
Mean validation score: 0.901928 (std: 0.007030)
Parameters: {'alpha': 1.2, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 2}
Model with rank: 2
Mean validation score: 0.901928 (std: 0.007030)
Parameters: {'alpha': 1.2, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 4}
Model with rank: 3
Mean validation score: 0.901928 (std: 0.007030)
Parameters: {'alpha': 1.2, 'kernel': DotProduct(sigma_0=1) + WhiteKernel(noise_level=1), 'n_restarts_optimizer': 3}
Wall time: 17min 51s (On charlotte)
4. "timeout or by a memory leak.", UserWarning
5. "timeout or by a memory leak.", UserWarning on Google Colaboratory.
"""


# %%
%%time
from sklearn.ensemble import ExtraTreesRegressor

estimator = ExtraTreesRegressor()
parameters = {
    'n_estimators': [10, 50, 100],
    'criterion': criterion,
    'max_depth': [None, 50, 100],
    'min_samples_split': [1, 2, 4, 8],
    'min_samples_leaf': [5e-1, 1, 1.5, 2],
    'min_weight_fraction_leaf': [0, 1e-1, 5e-1, 1, 1.5],
    'max_features': max_features,
    'max_leaf_nodes': [None, 1, 10, 20],
    'min_impurity_decrease': [0, 1e-1, 5e-1, 1],
    'min_impurity_split': [0, 1e-8, 1e-7, 1e-6, 1e-5],
    'bootstrap': bootstrap,
    'oob_score': oob_score,
    'random_state': random_state
    'warm_start': warm_start,
}

gscv_bggr = grid_search(estimator, parameters, X, y)

""" Results
1. 
"""


# %%
%%time
from sklearn.ensemble import GradientBoostingRegressor

estimator = GradientBoostingRegressor()
parameters = {
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'learning_rate': [0.2, 0.21, 0.22, 0.23, 0.24],
    'n_estimators': [30, 40, 50, 60, 70, 80],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [19, 20, 21, 22],
    'max_depth': [3, 4, 5, 6, 7]
}

gscv_gdbr = grid_search(estimator, parameters, X, y)

""" Results
1.
best params={'learning_rate': 0.31, 'loss': 'ls', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 21, 'n_estimators': 50}
2.
best params={'learning_rate': 0.3, 'loss': 'huber', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 20, 'n_estimators': 30}
3.
best params={'learning_rate': 0.28, 'loss': 'huber', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 20, 'n_estimators': 40}
4.
best score=0.8972064865825896
best params={'learning_rate': 0.22, 'loss': 'ls', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 21, 'n_estimators': 60}
"""

# %%
%%time
from sklearn.ensemble import HistGradientBoostingRegressor

estimator = HistGradientBoostingRegressor()
parameters = {
    'loss': ['least_squares'],
    'learning_rate': [1e-3, 1e-2, 1e-1, 5e-1, 1],
    'max_iter': [50, 100, 150],
    'max_leaf_nodes': [None, 1, 11, 21, 31, 41],
    'max_depth': [None, 1, 11],
    'min_samples_leaf': [10, 20, 30],
    'l2_regularization': [0],
    'max_bins': [256],
    'scoring': [None],
    'n_iter_no_change': [None],
    'tol': [1e-8, 1e-7, 1e-6],
    'random_state': random_state
}

gscv_hgdbr = grid_search(estimator, parameters, X, y)

""" Results
1.
""" 

# %%
%%time
from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor()
parameters = {
    'bootstrap': bootstrap,
    'n_estimators': [450, 500, 550, 600, 650],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [0.0005, 0.001, 0.005],
    'min_samples_leaf': [0.08, 0.09, 0.1, 0.11],
    'min_weight_fraction_leaf': [0.0, 0.1],
    'max_features': max_features,
    'min_impurity_decrease': [0, 0.1, 0.2],
}

gscv_rdft = grid_search(estimator, parameters, X, y)

""" Results
1.
best params={'bootstrap': True, 'max_depth': 6, 'max_features': 'sqrt', 'min_impurity_decrease': 0, 'min_samples_leaf': 0.1, 'min_samples_split': 0.01, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 700}
2.
best params={'bootstrap': True, 'max_depth': 5, 'max_features': 'sqrt', 'min_impurity_decrease': 0, 'min_samples_leaf': 0.09, 'min_samples_split': 0.005, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 650}
3.
best score=0.7205065255864675
best params={'bootstrap': True, 'max_depth': 4, 'max_features': 'auto', 'min_impurity_decrease': 0, 'min_samples_leaf': 0.08, 'min_samples_split': 0.0005, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 650}
"""


# %%
%%time
from sklearn.neural_network import MLPRegressor

estimator = MLPRegressor()
parameters = {
    'hidden_layer_sizes': [50, 100, 150],
    'activation': activation,
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.00001, 0.0001, 0.001],
    'batch_size': batch_size,
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'power_t': [0.4, 0.5, 0.6],
    'max_iter': [100, 200, 400],
    'shuffle': shuffle,
    'random_state': random_state,
    'tol': [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'warm_start': warm_start,
    'momentum': [0, 0.3, 0.6, 0.9, 1],
    'nesterovs_momentum': nesterovs_momentum,
    'early_stopping': early_stopping,
    'validation_fraction': [0.1],
    'beta_1': [0.3, 0.6, 0.9],
    'beta_2': [0.333, 0.666, 0.999],
    'epsilon': [1e-4, 1e-6, 1e-8, 1e-10],
    'n_iter_no_change': [1, 9, 10, 11, 20],
}

gscv_mlpr = grid_search(estimator, parameters, X, y)

""" Results
1.
"""

# %%
%%time
from xgboost import XGBRegressor

estimator = XGBRegressor()
parameters = {
    'booster': booster,
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.03],
    # 'verbosity': [0],
    # 'nthread': [],
    # 'disable_default_eval_metric': [0],
    # 'num_pbuffer': [],
    # 'num_feature': [],

}

gscv_xgbr = grid_search(estimator, parameters, X, y)

""" Results
1. sklearn_2019-10-16-08-19-55.gscv
Model with rank: 1
Mean validation score: 0.900128 (std: 0.010102)
Parameters: {'booster': 'gbtree'}
Model with rank: 2
Mean validation score: 0.900128 (std: 0.010102)
Parameters: {'booster': 'dart'}
Model with rank: 3
Mean validation score: -0.618425 (std: 0.110969)
Parameters: {'booster': 'gblinear'}
Wall time: 4.61 s
"""


# %%
%%time
from lightgbm import LGBMRegressor

estimator = LGBMRegressor()

# lgbm = lgb.LGBMRegressor(
#                             objective='regression',
#                             device='gpu',
#                             n_jobs=1,
#                         )
# param_dist = {'boosting_type': ['gbdt', 'dart', 'rf'],
#                 'num_leaves': sp.stats.randint(2, 1001),
#                 'subsample_for_bin': sp.stats.randint(10, 1001),
#                 'min_split_gain': sp.stats.uniform(0, 5.0),
#                 'min_child_weight': sp.stats.uniform(1e-6, 1e-2),
#                 'reg_alpha': sp.stats.uniform(0, 1e-2),
#                 'reg_lambda': sp.stats.uniform(0, 1e-2),
#                 'tree_learner': ['data', 'feature', 'serial', 'voting' ],
#                 'application': ['regression_l1', 'regression_l2', 'regression'],
#                 'bagging_freq': sp.stats.randint(1, 11),
#                 'bagging_fraction': sp.stats.uniform(loc=0.1, scale=0.9),
#                 'feature_fraction': sp.stats.uniform(loc=0.1, scale=0.9),
#                 'learning_rate': sp.stats.uniform(1e-6, 0.99),
#                 'max_depth': sp.stats.randint(1, 501),
#                 'n_estimators': sp.stats.randint(100, 20001),
#                 'gpu_use_dp': [True, False],
#                 }

parameters = {
    'boosting_type': boosting_type,
    'num_leaves': [21, 31, 41],
    'max_depth': [-1, 0, 1],
    'learning_rate': [1e-3, 1e-2, 1e-1, 1],
    'n_estimators': [0, 100, 200],
    'subsample_for_bin': [100000, 200000, 200000],
    # 'objective ': objective,
    # 'class_weight': [],
    'min_split_gain': [0, 1e-9, 1e-1, 1],
    'min_child_weight': [0, 1e-3, 1],
    'min_child_samples': [10, 20, 30],
    'subsample': [0, 1e-2, 1e-1, 1],
    'subsample_freq': [0, 1],
    'colsample_bytree': [0, 1e-1, 1],
    'reg_alpha': [0, 1e-1, 1],
    'reg_lambda': [0, 1e-1, 1],
    # 'random_state': random_state,
}

gscv_lgbn = grid_search(estimator, parameters, X, y)

""" Results
1. sklearn_2019-10-16-08-43-09.gscv
Model with rank: 1
Mean validation score: 0.889644 (std: 0.012444)
Parameters: {'boosting_type': 'gbdt'}
Model with rank: 2
Mean validation score: 0.000000 (std: 0.000000)
Parameters: {'boosting_type': 'rf'}
Model with rank: 3
Mean validation score: -8.525953 (std: 1.065686)
Parameters: {'boosting_type': 'dart'}
Wall time: 11min 8s
2. sklearn_2019-10-19-20-04-41.gscv
Model with rank: 1
Mean validation score: 0.949002 (std: 0.034350)
Parameters: {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 150, 'num_leaves': 20}
Model with rank: 1
Mean validation score: 0.949002 (std: 0.034350)
Parameters: {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 0, 'n_estimators': 150, 'num_leaves': 20}
Model with rank: 3
Mean validation score: 0.948998 (std: 0.034455)
Parameters: {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 150, 'num_leaves': 22}
Model with rank: 3
Mean validation score: 0.948998 (std: 0.034455)
Parameters: {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 0, 'n_estimators': 150, 'num_leaves': 22}
10028.4s
"""

# %%
estimators = [
    gscv_eln.best_estimator_,
    gscv_lss.best_estimator_,
    gscv_lsvr.best_estimator_,
    gscv_gusp.best_estimator_,
    gscv_xgbr.best_estimator_,
    gscv_lgbn.best_estimator_
]

estimators_for_vote = [
    ('elastic', gscv_eln.best_estimator_),
    ('lasso', gscv_lss.best_estimator_),
    ('lnsvr', gscv_lsvr.best_estimator_),
    ('gaus', gscv_gusp.best_estimator_),
    ('xgbr', gscv_xgbr.best_estimator_),
    ('lgbm', gscv_lgbn.best_estimator_)
]

# %%
%%time
from sklearn.ensemble import AdaBoostRegressor

estimator = AdaBoostRegressor()
parameters = {
    'base_estimator': estimators,
    'n_estimators': [50],
    'learning_rate': [0, 1],
    'loss': loss_for_ada,
    'random_state': random_state
}

gscv_adbr = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-13 06:05:34
Model with rank: 1
Mean validation score: 0.814 (std: 0.023)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False), 'learning_rate': 1, 'loss': 'exponential', 'n_estimators': 50, 'random_state': None}
Model with rank: 2
Mean validation score: 0.808 (std: 0.020)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False), 'learning_rate': 1, 'loss': 'linear', 'n_estimators': 50, 'random_state': None}
Model with rank: 3
Mean validation score: 0.618 (std: 0.078)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False), 'learning_rate': 1, 'loss': 'square', 'n_estimators': 50, 'random_state': None}
"""

# %%
%%time
from sklearn.ensemble import BaggingRegressor

estimator = BaggingRegressor()
parameters = {
    'base_estimator': estimators,
    'n_estimators': [10, 50],
    'max_samples': [5e-1, 1.0, 1.5, 2.0],
    'max_features': [5e-1, 1.0, 1.5, 2.0],
    'bootstrap': bootstrap,
    'bootstrap_features': bootstrap_features,
    'oob_score': oob_score,
    'warm_start': warm_start,
    'random_state': random_state
}

gscv_bggr = grid_search(estimator, parameters, X, y)

""" Results
1. 
"""

# %%
%%time
from math import inf
from sklearn.linear_model import RANSACRegressor

estimator = RANSACRegressor()
parameters = {
    'base_estimator': estimators,
    'min_samples': [0, 5e-1, 1],
    'residual_threshold': [None, 0, 1e-2, 1e-1, 1],
    'is_data_valid': [None],
    'is_model_valid': [None],
    'max_trials': [50, 100],
    'max_skips': [inf],
    'stop_n_inliers': [inf],
    'stop_score': [inf],
    'stop_probability': [0, 0.99, 1],
    'loss': loss_for_ransac,
    'random_state': random_state
}

gscv_ransac = grid_search(estimator, parameters, X, y)

""" Results
1. 2019-10-13 04:39:03
Model with rank: 1
Mean validation score: 0.316 (std: 0.221)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100,
               tol=1e-05, warm_start=False), 'is_data_valid': None, 'is_model_valid': None, 'loss': 'squared_loss', 'max_skips': inf, 'max_trials': 100, 'min_samples': 1, 'random_state': None, 'residual_threshold': 1, 'stop_n_inliers': inf, 'stop_probability': 1, 'stop_score': inf}
Model with rank: 2
Mean validation score: 0.310 (std: 0.244)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100,
               tol=1e-05, warm_start=False), 'is_data_valid': None, 'is_model_valid': None, 'loss': 'squared_loss', 'max_skips': inf, 'max_trials': 50, 'min_samples': 1, 'random_state': None, 'residual_threshold': 1, 'stop_n_inliers': inf, 'stop_probability': 1, 'stop_score': inf}
Model with rank: 3
Mean validation score: 0.309 (std: 0.212)
Parameters: {'base_estimator': HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100,
               tol=1e-05, warm_start=False), 'is_data_valid': None, 'is_model_valid': None, 'loss': 'absolute_loss', 'max_skips': inf, 'max_trials': 100, 'min_samples': 1, 'random_state': None, 'residual_threshold': 1, 'stop_n_inliers': inf, 'stop_probability': 0.99, 'stop_score': inf}
"""


# %%
from sklearn.ensemble import VotingRegressor

value = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

weights = itertools.product(value, value, value)
weights = np.array(list(weights))

# vt = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3), ('lgr', reg4)])
vt = VotingRegressor(estimators_for_vote)
param_vt = {
    'weights': weights
}

estimator = vt
parameters = param_vt

rscv_vt = RandomizedSearchCV(estimator, parameters, verbose=1000, n_jobs=-1, cv=4)
rscv_vt.fit(X, y)

print("best score={}".format(rscv_vt.best_score_))
print("best params={}".format(rscv_vt.best_params_))

dump(rscv_vt, 'rscv_vt.rscv', compress=True)

""" Results
1.
"""

# %%
best_estimator = rscv_vt.best_estimator_
best_estimator.fit(X, y)


# %%
preds = estimator.predict(test)

# %%
# Submission
name = 'submission_{}.csv'.format(now_for_file())
rw.submit_prediction(name, list(range(1461, 2920)), np.expm1(preds))

