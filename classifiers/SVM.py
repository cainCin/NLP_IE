from sklearn.svm import SVC

class CLASSIFIER:
    def __init__(self):
        self.load_default_classifier()
        pass
    
    def load_default_classifier(self):
        self.load_svm()
        
    def load_svm(self, kernel='poly', degree=8):
        self.model = SVC(kernel=kernel, degree=degree)
        
    def fit(self, X_train, y_train):
        self.model.probability = True
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X):
        return self.model._predict_proba(X)
    
    def score_log(self, X):
        return self.model._predict_log_proba(X)
    
    def predict_MC(self, X):
        self.decision_function_shape='ovr'
        return self.model.decision_function(X)