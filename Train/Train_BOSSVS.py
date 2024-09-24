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


def train(model = 'TSF', classifier = None, parameters = None, threshold = 500):

    with open('/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data.pickle', 'rb') as file:
        series = pickle.load(file)
        labels = pickle.load(file)

    y = np.where(labels[:, 2] >= threshold, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(series, y, random_state = 0, test_size = 0.2)

    
    clf = GridSearchCV(classifier, parameters, scoring = 'f1_macro', cv = 5, return_train_score = True)
    clf.fit(series, y)


    pd.DataFrame(clf.cv_results_).to_csv(f"{model}_gridsearch_result.csv")
    print(clf.best_params_)
    print('Train Score: ',clf.score(X_train, y_train))
    print('Test score', clf.score(X_test, y_test))

    

if __name__ == "__main__":
    
    parameters = {'window_size': [10, 15, 20, 25, 30], 'n_bins':[2, 4, 6, 8], 'word_size': [2, 4, 6, 8]}
    model = BOSSVS()
    train(model = 'BOSSVS', classifier = model, parameters = parameters, threshold = 500)