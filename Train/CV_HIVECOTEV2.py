# Basic libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from Measurements import HSS2, TSS

# import models
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2

def HSS_caller(y_true, y_pred):
    table = confusion_matrix(y_true, y_pred)
    return HSS2(table.ravel())

def TSS_caller(y_true, y_pred):
    table = confusion_matrix(y_true, y_pred)
    return TSS(table.ravel())

def train(model = 'TSF', classifier = None, threshold = 500, window_st = '10', window_end = '12'):

    with open(f'/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data_st{window_st}end{window_end}.pickle', 'rb') as file:
        series = pickle.load(file)
        labels = pickle.load(file)

    y = np.where(labels[:, 2] >= threshold, 1, 0)
    
    scoring_fn = {
        'Precision': 'precision',
        'HSS': 'HSS_caller',
        'TSS': 'TSS_caller',
        'F1_macro': 'f1_macro'
    }

    scores = cross_validate(classifier, series, y, cv = 5, scoring=scoring_fn, return_train_score=True, return_estimator=True)

    with open(f'{model}_CV_estimator_st{window_st}end{window_end}.pickle', 'wb') as file:
        pickle.dump(scores, file)

    print('Train Score: ', scores['train_score'])
    print('Test score', scores['test_score'])

    

if __name__ == "__main__":
    
    model = HIVECOTEV2()
    train(model = 'HIVECOTEV2', classifier = model, threshold = 500, window_st = '20', window_end = '25')