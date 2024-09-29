import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train(model = 'Logistic', classifier = None, threshold = 500):

    df = pd.read_csv("/media/jh/maxone/Research/GSU/Research1_xray_flux/MultiwayIntegration_2010_to_2018_conf_rxfi.csv")
    df = df.loc[(df['cme_vel'].notnull()) & (df['Background_X-ray_flux'].notnull()) 
                 & (df['goes_class_num'].notnull()) & (df['fluorescence'] >= 0) 
                 & (df['rise_gradient'] >= 0),  :]


    y = np.where(df['cme_vel'] >= threshold, 1, 0)
    features = df[['goes_class_num', 'Background_X-ray_flux' ,'relative_X-ray_flux_increase', 
                   'fluorescence', 'Flare_rise_time', 'Flare_decay_time', 'Flare_duration', 
                   'rise_gradient', 'decay_gradient', 'avg_fluorescence', 'avg_rise_fluorescence', 
                   'avg_decay_fluorescence']] #'fluorescence_fhalf', 'fluorescence_lhalf',
    
    features = features.astype('float32')
    for column in features.columns: 
        features[column] = (features[column]-features[column].min())  / (features[column].max() - features[column].min())  
    # print(features)

    scores = cross_validate(classifier, features, y, cv = 5, scoring='f1_macro', return_train_score=True, return_estimator=True)
    
    with open(f'{model}_CV_estimator.pickle', 'wb') as file:
        pickle.dump(scores, file)

    print('Train Score: ', scores['train_score'])
    print('Test score', scores['test_score'])


if __name__ == "__main__":
    
    model = LogisticRegression(random_state=7, C=1000, class_weight={0: 0.4, 1:0.6}, max_iter=5000, penalty='l1', solver='liblinear')
    train(model = 'LogisticReg', classifier=model, threshold=500)