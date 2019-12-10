# Utility functions
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define
path = '../input/home-data-for-ml-course'
VERBOSE_ = 1
N_JOBS_ = -1
CV_ = 4
ERROR_SCORE_ = 0.0


def now_str():
    now = datetime.datetime.now()
    return "{0:%Y-%m-%d %H:%M:%S}".format(now)


def now_for_file():
    now = datetime.datetime.now()
    return "{0:%Y-%m-%d-%H-%M-%S}".format(now)


def duplicate_count(arry):
    # https://stackoverflow.com/questions/44479052/python-find-the-number-of-duplicates-in-a-string-text
    # elem is an array of the unique elements in a string
    # and count is its corresponding frequency
    elem, count = np.unique(arry, return_counts=True)

    return np.sum(count>1)


def report(gridscv, n_top=3):
    results = gridscv.cv_results_
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.6f} (std: {1:.6f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))


def scatter(preds, y):
    plt.style.use('dark_background')
    plt.scatter(y, preds, marker='o', color='orange', alpha=0.5)
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.show()


def submit_prediction(scv, test):
    name = 'submission_{}_{}.csv'.format(scv.estimator.__module__.split('.')[-1].lower(), now_for_file())
    output = pd.DataFrame({'Id': list(range(1461, 2920)), 'SalePrice': np.expm1(scv.predict(test))})
    output.to_csv(name, index=False)
