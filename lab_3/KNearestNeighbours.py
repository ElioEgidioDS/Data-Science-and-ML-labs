import numpy as np
from collections import Counter, defaultdict

def _majority_voting(votes):
    count = Counter(votes)
    return count.most_common(1)[0][0]

def _weight_majority_voting(votes, weights):
    #if i access non existent key it adds it and doesnt go into error
    count = defaultdict(lambda: 0)
    for vote, weight in zip(votes, weights):
        count[vote] += weight
    
    return max(count.items(), key=lambda x: x[1])[0]
    '''
    return max(count.items(), key=lambda x: x[1])[0]

    count.items() is an iterable of (label, total_weight) pairs.

    max(..., key=lambda x: x[1]) finds the pair with the largest total_weight.

    [0] extracts the label from the (label, total_weight) pair.

    So the function returns the label with the largest summed weight
    '''

class KNearestNeighbors :
    def __init__ ( self , k , distance_metric = " euclidean ", weights = "uniform" ) :
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
    def fit( self , X , y ) :
        """
        Store the ' prior knowledge ' of you model that will be used
        to predict new labels .
        :param X : input data points , ndarray , shape = (R , C ) .
        :param y : input labels , ndarray , shape = (R ,) .
        """
        #just storing the dataset, so we can do predictions on it
        self.x_train = X
        self.y_train = y

        self.X_train_reshaped = np.expand_dims(self.x_train, 1)
        self.X_train_norm = ((self.x_train**2).sum(axis=1)**.5).reshape(-1,1)

        pass # TODO : implement it !
    def predict ( self , X_test ) :
        """ Run the KNN classification on X .
        : param X : input data points , ndarray , shape = (N , C ) .
        : return : labels : ndarray , shape = (N ,) .
        """

        match self.distance_metric:
            case "euclidean":
                dist_matrix = self._euclidean_distance(X_test)
            case "cosine":
                dist_matrix = self._cosine(X_test)
            case "manhattan":
                dist_matrix = self._manhattan(X_test)     
            case _:
                return "error"

        knn = dist_matrix.argsort(axis=0)[:self.k,:].T

        if self.weights == "uniform":
            ypred = np.array([_majority_voting(self.y_train[knn][i]) for i in range(len(self.y_train[knn]))])
        elif self.weights == "distance":
            weights = 1/(np.take_along_axis(dist_matrix,knn.T,0)+1e-5)
            ypred = np.array([_weight_majority_voting(self.y_train[knn][i], weights[:,i]) for i in range(len(self.y_train[knn])) ])
        return ypred

    #keeps it a private helper function
    def _euclidean_distance(self, X_test):
        X_diff = self.X_train_reshaped - X_test
        dist_matrix = ((X_diff**2).sum(axis=2))**.5
        return dist_matrix
    
    def _cosine(self, X_test):
        X_test_norm = ((X_test**2).sum(axis=1)**.5).T
        dot_prods = self.x_train @ X_test.T
        dist_matrix = 1 - abs(dot_prods / self.X_train_norm.reshape(-1,1) / X_test_norm)
        return dist_matrix

    def _manhattan(self, X_test):
        self.X_test_reshaped = np.expand_dims(X_test, 1)
        X_diff = self.X_train_reshaped - X_test
        dist_matrix = abs(X_diff).sum(axis=2)
        return dist_matrix


        