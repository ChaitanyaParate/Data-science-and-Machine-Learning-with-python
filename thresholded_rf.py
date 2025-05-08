from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class ThresholdedRandomForest:
    def __init__(self,thresh_hold = 0.5, **kwargs):
        self.thresh_hold = thresh_hold
        self.model = RandomForestClassifier(**kwargs)

    def fit(self,X,y):
        return self.model.fit(X,y)
        
    def predict(self,X_test):
        y_cust_prob = self.model.predict_proba(X_test)[:, 1]
        return (y_cust_prob >= self.thresh_hold).astype(int)
    
    def predict_proba(self,X_test):
        return self.model.predict_proba(X_test)[:, 1]
    
    def score(self,X,y):
        from sklearn.metrics import accuracy_score, recall_score
        return accuracy_score(y, self.predict(X)), recall_score(y, self.predict(X))

def k_fold(x,y,fold=1):
    x_train_folds = np.array_split(x, fold)
    y_train_folds = np.array_split(y, fold)

    out = {}

    n_est = [10, 50, 200, 500, 94, 700]
    max_dep = [5, 7, 9]

    for m in max_dep:
        out[m] = {}
        for n in n_est:
            out[m][n] = []

    for f in range(fold):

        X_val = x_train_folds[f]
        y_val = y_train_folds[f]

        X_train_cv = np.vstack([fold_ for z, fold_ in enumerate(x_train_folds) if z != f])
        y_train_cv = np.hstack([fold_ for z, fold_ in enumerate(y_train_folds) if z != f])

        for i in max_dep:
            for a in n_est:
                clf = RandomForestClassifier(
                n_estimators=a,
                max_depth=i,
                max_features='sqrt',
                random_state=0
                )

                clf.fit(X_train_cv, y_train_cv)
                y_pre = clf.predict(X_val)
                accuracy = accuracy_score(y_val,y_pre)

                out[i][a].append(accuracy)

    return out


    
