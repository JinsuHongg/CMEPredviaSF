# Basic libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# import models
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2

def train(model = 'TSF', classifier = None, parameters = None, threshold = 500, window_st = '10', window_end = '12'):

    with open(f'/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data_st{window_st}end{window_end}.pickle', 'rb') as file:
        series = pickle.load(file)
        labels = pickle.load(file)

    y = np.where(labels[:, 2] >= threshold, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(series, y, random_state = 0, test_size = 0.2)

    
    clf = GridSearchCV(classifier, parameters, scoring = 'f1_macro', cv = 5, return_train_score = True)
    clf.fit(series, y)


    pd.DataFrame(clf.cv_results_).to_csv(f"{model}_gridsearch_result.csv", index = False)
    print(clf.best_params_)
    print('Train Score: ',clf.score(X_train, y_train))
    print('Test score', clf.score(X_test, y_test))

    
if __name__ == "__main__":
    
    parameters = {'tsf_params':[100, 300],
                  'rise_params': [100, 300]} #, 'time_limit_in_minutes': [0, 1], 'stc_params': [1, 2, 3],
    
    hc2 = HIVECOTEV1()
    train(model = 'HIVECOTEV1', classifier = hc2, parameters = parameters, threshold = 500, window_st = '20', window_end = '25')