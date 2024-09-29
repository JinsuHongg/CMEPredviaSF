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


def train(model = 'TSF', classifier = None, parameters = None, threshold = 500):

    with open('/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data.pickle', 'rb') as file:
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
    
    parameters = {'window_size': [0.2, 0.4, 0.6, 0.8], 
                  'word_size':[0.2, 0.4, 0.6, 0.8], 
                  'strategy':['uniform', 'quantile', 'normal'],
                  'n_bins': [4, 8, 12],
                  'window_step': [0.4, 0.6, 0.8, 1],
                  'threshold_std': [0.01, 0.1, 0.001] }
    model = SAXVSM()
    train(model = 'SAXVSM', classifier = model, parameters = parameters, threshold = 500)