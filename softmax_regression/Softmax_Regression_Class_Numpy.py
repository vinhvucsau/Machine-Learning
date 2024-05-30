import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegressor():
    def __init__(self,random_state , epochs,learning_rate):
        self.theta = None
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
    @staticmethod
    def one_hot(y,k):
        y = y.astype(int)
        one_hot = np.zeros((len(y),k))
        one_hot[np.arange(len(y)),y] = 1
        return one_hot
    
    def predict(self,X):
        z = np.dot(X,self.theta)
        z = np.exp(z)
        y_pred = z / np.sum(z,axis=1).reshape((-1,1))
        return y_pred
    
    def compute_loss(self,y,y_pred):
        loss = np.sum(-np.log(np.sum(y * y_pred,axis=1))) / len(y)
        return loss
    
    def compute_gradient(self,X,y,y_pred):
        return np.dot(X.T,(y_pred-y))
    
    def fit(self,X,y):
        np.random.seed(self.random_state)
        k = len(set(y))
        y_one_hot = self.one_hot(y,k)
        X_new = np.hstack([np.ones((len(X),1)),X])
        self.theta = np.random.random((len(X_new[0]),k))
        losses = []
        for epoch in range(self.epochs):
            y_hat = self.predict(X_new)
            loss = self.compute_loss(y_one_hot,y_hat)
            gradient = self.compute_gradient(X_new,y_one_hot,y_hat)
            self.theta = self.theta - self.learning_rate * gradient
            losses.append(loss)
        return losses
    


# load data 
df = pd.read_csv('.\datasets\iris_2D_3c.csv')
df = df.values

# using softmax_regression
epochs = 100
softmax = SoftmaxRegressor(learning_rate=0.02,epochs=epochs,random_state=42)
losses = softmax.fit(df[:,:2],df[:,2])

# Visualization 
plt.figure(figsize=(8, 6))
plt.plot(np.arange(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Softmax Regression Training Loss')
plt.grid(True)
plt.show()
