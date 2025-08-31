import numpy as np
import pandas as pd

#Hjelpefunksjoner
def sigmoid_lg(theta, X):
   return 1/(1 + np.exp( -np.matmul(X, theta) ))

def compute_gradient(theta, X, y):
    y_pred = sigmoid_lg(theta, X)

    return 1/y.shape[0] * ( np.matmul(np.transpose(X), y_pred - y))

class LogisticRegression():

    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 100
        self.weights = None
        self.losses, self.train_accuracies = [], []

    def compute_loss_lg(self, y, y_pred):
        pass

    def fit_lg(self, trainData):
        #Forandrer formatet på dataen til numpy
        X = trainData[['x0', 'x1']].to_numpy()
        y = trainData[['y']].to_numpy()

        #X kommer uten biaskolonne så legger til det
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

        self.weights = np.zeros((X.shape[1], 1))

        theta = self.weights

        for i in range(self.epochs):
            theta = theta - self.learning_rate*compute_gradient(theta, X, y)
        self.weights = theta


    def predict_lg(self, testData):
        X = testData[['x0', 'x1']].to_numpy()
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

        y_pred = sigmoid_lg(self.weights, X) #GIR VELDIG SMÅ VERDIER. NESTEN SÅ DEN IKKE PRØVER
        print(y_pred)
        return [1 if _y > 0.5 else 0 for _y in y_pred.flatten()]