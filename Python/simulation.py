#!/usr/bin/env python
# coding: utf-8

from joblib import load, dump
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utility import now_for_file, report, scatter

# Parameter of cross validation
VERBOSE_ = 1
N_JOBS_ = -1
CV_ = 4
ERROR_SCORE_ = 0.0
N_ITER_SEARCH_ = 20


def dump_estimator(estimator):
    # Save computed estimator
    name = estimator.estimator.__module__.split('.')[-1].lower()
    time = now_for_file()
    dump(estimator, '{}_{}.estm'.format(name, time), compress=True)

    return name


def current_best_dt(X, y, tree_size):
    # Define estimator. Specify a number for random_state to ensure same results each run
    estimator = DecisionTreeRegressor(max_leaf_nodes=tree_size, random_state=0)
    # Fit estimator
    estimator.fit(X, y)

    return estimator


def get_mae_dt(max_leaf_nodes, train_X, val_X, train_y, val_y):
    estimator = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    estimator.fit(train_X, train_y)
    preds_val = estimator.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)

    # print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

    return(mae)


def current_best_rf(X, y, tree_size):
    # Define estimator. Specify a number for random_state to ensure same results each run
    estimator = RandomForestRegressor(max_leaf_nodes=tree_size, random_state=1)
    # Fit estimator
    estimator.fit(X, y)

    return estimator


def get_mae_rf(max_leaf_nodes, train_X, val_X, train_y, val_y):
    estimator = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    estimator.fit(train_X, train_y)
    preds_val = estimator.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)

    # print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

    return(mae)


def get_best_tree_size(X, y, tree):
    if tree == 'dt':
        get_mae = get_mae_dt
    elif tree == 'rf':
        get_mae = get_mae_rf
    else:
        raise

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    candidate_max_leaf_nodes = np.arange(10, 2000, 10)

    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    scores = {leaf_size: get_mae_dt(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

    return min(scores, key=scores.get)



def estimator_cv(estimator, param, X, y):
    estimatorCV.set_params(param)
    estimatorCV.fit(X, y)

    print('{}_{}.ecv'.format(name, time))
    report(estimatorCV)

    preds = estimatorCV.predict(X)
    print('Mean Absolute Error: {}'.format(mean_absolute_error(np.exp(y), np.exp(preds))))
    scatter(preds, y)

    return gridscv


def current_best_elastic(X, y):
    estimatorCV = linear_model.ElasticNetCV()
    parameters = {
        'alpha': [0.0011],
        'l1_ratio': [0.008],
        # 'fit_intercept': fit_intercept,
        # 'normalize': normalize,
        'max_iter': [1050],
        # 'warm_start': warm_start,
        'positive': [True],
        'precompute': [True],
        'selection': ['random'],
        # 'random_state': random_state,
        'tol': [0.050],
        'cv': CV_,
        'verbose': VERBOSE_,
        'n_jobs': N_JOBS_
    }
    estimator_cv(estimatorCV, parameters, X, y)

    return estimatorCV


def current_best_lasso(X, y):
    estimatorCV = linear_model.LassoLarsCV()
    parameters = {
        'alpha': [5e-6, 1e-5, 6e-5, 5e-5, 4e-5, 2e-5, 1e-4, 8e-4, 6e-5, 5e-4, 1e-3],
        'max_iter': [800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050],
        'normalize': normalize,
        'positive': positive,
        'precompute': [True],
        'selection': ['random'],
        # 'fit_intercept': fit_intercept,
        'tol': [1e-3, 2e-3, 1e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-1, 9e-1, 8e-1, 6e-1, 5e-1],
        'warm_start': warm_start,
        # 'random_state': random_state,
    }
    estimatorCV.set_params(parameters)
    estimatorCV.fit(X, y)

    return estimatorCV


def current_best_ridge(X, y):
    estimator = linear_model.RidgeCV(cv=10)
    estimator.fit(X, y)

    return estimator


def current_best_AdaBoost(X, y, tree_size):
    estimator = AdaBoostRegressor(DecisionTreeRegressor(max_leaf_nodes=tree_size), random_state=0)
    estimator.fit(X, y)

    return estimator


def predict_lasso(X, y):
    y_pred = np.exp(lasso.predict(X))
    mae = error(y, y_pred)

    return y_pred, mae


def predict_ridge(X, y):
    y_pred = np.exp(ridge.predict(X))
    mae = error(y, y_pred)

    return y_pred, mae


def current_best_xgboost(X, y):
    estimator = XGBRegressor()
    estimator.fit(X, y)

    return estimator


def random_search(estimator, param, X, y, n_iter_search=N_ITER_SEARCH_):
    randomscv = RandomizedSearchCV(estimator, param_distributions=param, n_iter=n_iter_search, n_jobs=N_JOBS_, cv=CV_, verbose=VERBOSE_, error_score=ERROR_SCORE_)
    randomscv.fit(X, y)

    # Save computed estimator
    name = dump_estimator(randomscv)

    # Report
    print('{}'.format(name))
    report(randomscv)
    preds = randomscv.predict(X)
    print('Mean Absolute Error: {}'.format(mean_absolute_error(np.exp(y), np.exp(preds))))

    # Plot
    scatter(preds, y)

    return randomscv


def grid_search(estimator, param, X, y):
    gridscv = GridSearchCV(estimator, param, n_jobs=N_JOBS_, cv=CV_, verbose=VERBOSE_, error_score=ERROR_SCORE_)
    gridscv.fit(X, y)

    # Save computed estimator
    name = dump_estimator(gridscv)

    # Report
    print('{}'.format(name))
    report(gridscv)
    preds = gridscv.predict(X)
    print('Mean Absolute Error: {}'.format(mean_absolute_error(np.exp(y), np.exp(preds))))

    # Plot
    scatter(preds, y)

    return gridscv

