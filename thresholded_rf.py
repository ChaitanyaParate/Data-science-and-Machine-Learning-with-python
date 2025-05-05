from sklearn.ensemble import RandomForestClassifier

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
    