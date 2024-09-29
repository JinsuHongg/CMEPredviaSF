# Basic libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# import models
from pyts.classification import TimeSeriesForest
from pyts.classification import LearningShapelets
from pyts.classification import BOSSVS
from pyts.classification import SAXVSM

def train(model = 'TSF', classifier = None, threshold = 500, window_st = '10', window_end = '12'):

    with open(f'/home/jh/2python_pr/CMEPredviaSF/Dataset_Creation/Timeseries_data_st{window_st}end{window_end}.pickle', 'rb') as file:
        series = pickle.load(file)
        labels = pickle.load(file)

    y = np.where(labels[:, 2] >= threshold, 1, 0)
    scores = cross_validate(classifier, series, y, cv = 5, scoring='f1_macro', return_train_score=True, return_estimator=True)

    with open(f'{model}_CV_estimator_st{window_st}end{window_end}.pickle', 'wb') as file:
        pickle.dump(scores, file)

    print('Train Score: ', scores['train_score'])
    print('Test score', scores['test_score'])

    

if __name__ == "__main__":
    
    model = TimeSeriesForest(random_state = 7, max_depth = 5, class_weight={0:0.4, 1:0.6})
    train(model = 'TSF', classifier = model, threshold = 500, window_st = '10', window_end = '12')