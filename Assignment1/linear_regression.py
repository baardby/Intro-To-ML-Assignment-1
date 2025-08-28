import numpy as np

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = 0.001
        self.epochs = 5000
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
        m = y.shape[0]
        bias = np.ones((m, 1))
        X_data = np.hstack((bias, np.array(X).reshape((m, 1)))) #Endret også slik at de får riktige dimensjoner til numpy regning
        y_data = np.array(y).reshape((m, 1)) #Samme her også
        
        self.weights = np.zeros((X_data.shape[1], 1)) #Initialisering
        theta = self.weights
        
        for i in range(self.epochs):
            gradient = 2/m * np.matmul(np.transpose(X_data), np.matmul(X_data, theta) - y_data)
            theta = theta - self.learning_rate*gradient
        self.weights = theta

        #Neste forsøk er å dele datasettet i to og se om den kan klare det fremdeles da!

    
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
        X_data = np.array(X).reshape((X.shape[0], 1))
        print(self.weights[0])
        y_pred = np.matmul(X_data, self.weights[1]) + self.weights[0]
        return y_pred





