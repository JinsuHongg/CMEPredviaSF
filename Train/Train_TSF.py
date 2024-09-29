# Basic libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# import models
from pyts.classification import TimeSeriesForest
from pyts.classification import LearningShapelets
from pyts.classification import BOSSVS
from pyts.classification import SAXVSM

def train(model = 'TSF', classifier = None, parameters = None, threshold = 500, window_st = '10', window_end = '12'):

    with open(f'/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data_st{window_st}end{window_end}.pickle', 'rb') as file:
        series = pickle.load(file)
        labels = pickle.load(file)

    y = np.where(labels[:, 2] >= threshold, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(series, y, random_state = 0, test_size = 0.2)

    clf = GridSearchCV(classifier, parameters, scoring = 'f1_macro', cv = 5, return_train_score = True)
    clf.fit(series, y)

    pd.DataFrame(clf.cv_results_).to_csv(f"{model}_gridsearch_result_st{window_st}end{window_end}.csv", index = False)
    print(clf.best_params_)
    print(clf.best_score_)

if __name__ == "__main__":
    
    parameters = {'class_weight': [{0:0.6, 1:0.4}, {0:0.5, 1:0.5}, {0:0.4, 1:0.6}], 
                  'max_depth':[3, 5, 7],
                  'n_estimators': [100, 300, 500, 700]}
    model = TimeSeriesForest(random_state = 7)
    train(classifier = model, parameters = parameters, threshold = 500, window_st = '20', window_end = '12')