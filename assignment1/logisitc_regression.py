import numpy as np
import pandas as pd

#Hjelpefunksjoner
def sigmoid_lg(theta, X):
   return 1/(1 + np.exp( -np.matmul(X, theta) ))

def compute_gradient(theta, X, y):
    y_pred = sigmoid_lg(theta, X)

    return 1/y.shape[0] * ( np.matmul(np.transpose(X), y_pred - y) )

class LogisticRegression():

    def __init__(self):
        self.learning_rate = 0.1
        self.epochs = 1000
        self.weights = None
        self.losses, self.train_accuracies = [], []

    def compute_loss_lg(self, y, y_pred):
        pass

    def fit_lg(self, trainData, alteredData):
        #Forandrer formatet på dataen til numpy
        y = trainData[['y']].to_numpy()

        #if setning for å prøve forskjellige måter å justere dataen på
        if alteredData == 0: #Uten justeringer
            #X kommer uten biaskolonne så legger til det
            X = trainData[['x0', 'x1']].to_numpy()
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))
        elif alteredData == 1: #Kun en feature: x0*x1
            x0 = trainData['x0'].to_numpy()
            x1 = trainData['x1'].to_numpy()
            X = x0*x1
            X = X.reshape((y.shape[0], 1))
        elif alteredData == 2: #Tre features: x0, x1 og x0*x1
            X = trainData[['x0', 'x1']].to_numpy()
            dummy = X[:, 0] * X[:, 1]
            dummy = dummy.reshape((y.shape[0], 1))
            X = np.hstack((X, dummy))
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))

        self.weights = np.ones((X.shape[1], 1))

        theta = self.weights

        for i in range(self.epochs):
            theta = theta - self.learning_rate*compute_gradient(theta, X, y)
        self.weights = theta


    def predict_lg(self, testData, alteredData):
        
        if alteredData == 0: #Ujustert data
            X = testData[['x0', 'x1']].to_numpy()
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))
        elif alteredData == 1: #Kun en feature: x0*x1
            x0 = testData['x0'].to_numpy()
            x1 = testData['x1'].to_numpy()
            X = x0*x1
            X = X.reshape((x0.shape[0], 1))
        elif alteredData == 2: #Tre features: x0, x1 og x0*x1
            X = testData[['x0', 'x1']].to_numpy()
            dummy = X[:, 0] * X[:, 1]
            dummy = dummy.reshape((X.shape[0], 1))
            X = np.hstack((X, dummy))
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))

        y_pred = sigmoid_lg(self.weights, X)
        return [1 if _y > 0.5 else 0 for _y in y_pred.flatten()]