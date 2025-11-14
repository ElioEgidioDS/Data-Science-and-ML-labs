from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MyRandomForestClassifier:
    def __init__(self, n_estimators=10, max_features='sqrt'):
        self.trees = [ DecisionTreeClassifier(max_features=max_features)\
                       for _ in range(n_estimators) ]
    
    def fit(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        for tree in self.trees:
            subset = np.random.choice(range(X.shape[0]),
                                      size=X.shape[0],
                                      replace=True)
            tree.fit(X[subset], y[subset])
    
    def predict(self, X):
        X = X.to_numpy()
        predictions = [ tree.predict(X) for tree in self.trees ] 
        return mode(predictions, axis=0)[0][0]