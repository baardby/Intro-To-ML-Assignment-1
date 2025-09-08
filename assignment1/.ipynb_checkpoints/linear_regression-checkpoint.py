import numpy as np

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = 0.05
        self.epochs = 100
        self.weights = None
        self.losses, self.train_accuracies = [], []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        # Dataen kommer uten bias så vi legger til en bias kolonne
        m = y.shape
        bias = np.transpose(np.ones(m))
        X_data = np.hstack((bias, X))
        
        self.weights = np.zeros(X_data[0].shape) #Initialisering
        
        theta = np.transpose(self.weights)
        return X_data[0].shape
        
        #for i in range(self.epochs):
            #np.matmul(X_data, np.array([[1],[2]])) - y
            #gradient = 2/m * np.matmul(np.transpose(X_data), np.matmul(X_data, theta) - y)
            #theta = theta - self.learning_rate*gradient
        #self.weights = np.transpose(theta) 
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        y_pred = np.matmul(X, self.weights[1:]) + self.weights[0]
        return y_pred





