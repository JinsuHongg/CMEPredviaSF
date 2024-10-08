import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train(model = 'Logistic', classifier = None, parameters = None, threshold = 500):

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
    
    X_train, X_test, y_train, y_test = train_test_split(features, y, random_state = 0, test_size = 0.2)

    clf = GridSearchCV(classifier, parameters, scoring = 'f1_macro', cv = 5, return_train_score = True)
    clf.fit(features, y)

    pd.DataFrame(clf.cv_results_).to_csv(f"{model}_gridsearch_result.csv", index = False)
    print(clf.best_params_)
    print('Train Score: ',clf.score(X_train, y_train))
    print('Test score', clf.score(X_test, y_test))


if __name__ == "__main__":
    
    parameters = {'class_weight': [{0:0.7, 1:0.3}, {0:0.6, 1:0.4}, {0:0.5, 1:0.5}, {0:0.4, 1:0.6}, {0:0.3, 1:0.7}],
                  'penalty': ['l2', None],  # e
                  'C':[1e5, 1e4, 5e3, 1e3, 100, 10, 1], 'solver': ['saga'], # 'lbfgs', 'newton-cg
                  'max_iter':[100, 1000, 10000]} 
    
    model = LogisticRegression(random_state = 7)
    train(model = 'LogisticReg', classifier = model, parameters = parameters, threshold = 500)