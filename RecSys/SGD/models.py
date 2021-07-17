import numpy as np
import time
import argparse
from sklearn.metrics import mean_squared_error
import dataset

class SGD_basic():
    def __init__(self, R, factor, learning_rate, lambda_param, epochs):
        """
        :param R: rating Matrix
        :param factor: number of latent factor
        :lambda_param: for regularization
        :epochs: how many update
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._factor = factor
        self._learning_rate = learning_rate
        self._lambda_param = lambda_param
        self._epochs = epochs

    def fit(self):
        #init latent factors
        np.random.seed(0)
        self._U = np.random.normal(size=(self._num_users, self._factor))
        self._V = np.random.normal(size=(self._num_items, self._factor))
        self._non_zeros = [ (i, j, self._R[i, j]) for i in range(self._num_users)
                  for j in range(self._num_items) if self._R[i, j] > 0 ]

        t = time.time()
        for epoch in range(self._epochs):
            if epoch ==0:
                print("기존 loss:", mean_squared_error(self._R, np.dot(self._U,self._V.T)))
            
            for i,j,r in self._non_zeros:
                self.gradient_descent(i,j, r)
            
            rmse = self.calculate_rmse()            
            if epoch % 10 ==0:
                print("Iteration: %d, loss = %.6f, time : %.4f" % (epoch+1, rmse, time.time()-t))

    def gradient(self, error, u, i):
        """
        gradient of latent feature for GD
        param error : rating - prediction error
        param u : user index
        param i : item index
        """
        
        dp = (error * self._V[i, :]) - (self._lambda_param * self._U[u, :])
        dq = (error * self._U[u, :]) - (self._lambda_param * self._V[i, :])
    
        return dp, dq

    def gradient_descent(self, i, j, r):
             
        prediction = np.dot(self._U[i], self._V[j])
        error = r - prediction
        
        dp, dq = self.gradient(error, i, j)
        self._U[i, :] += self._learning_rate * dp
        self._V[j, :] += self._learning_rate * dq
        
    # 실제 R 행렬과 예측 행렬의 오차를 구하는 함수
    def calculate_rmse(self):
        error = 0

        xi, yi = self._R.nonzero() 
        predicted = np.dot(self._U,self._V.T)

        for x, y in zip(xi, yi):
            error += pow(self._R[x, y] - predicted[x, y], 2) 

        return np.sqrt(error/len(xi))