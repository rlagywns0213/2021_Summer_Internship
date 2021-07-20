import numpy as np
import time
import argparse
from sklearn.metrics import mean_squared_error
import dataset

class Matrix_Factorization_bias():
    def __init__(self, R, R_test, factor, learning_rate, lambda_param, epochs):
        """
        :param R: rating Matrix
        :param factor: number of latent factor
        :lambda_param: for regularization
        :epochs: how many update
        """
        self._R = R
        self._R_test = R_test
        self._num_users, self._num_items = R.shape
        self._factor = factor
        self._learning_rate = learning_rate
        self._lambda_param = lambda_param
        self._epochs = epochs
        self._p = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(R), dtype = np.float64) #binary matrix

    def fit(self):
        #init latent factors
        np.random.seed(0)
        self._U = np.random.normal(scale = 1.0/self._factor,size=(self._num_users, self._factor))
        self._V = np.random.normal(scale = 1.0/self._factor, size=(self._num_items, self._factor))
        self._non_zeros = [ (i, j, self._R[i, j]) for i in range(self._num_users)
                  for j in range(self._num_items) if self._R[i, j] > 0 ]

        # init biases
        self._b_U = np.zeros(self._num_users)
        self._b_V = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)]) # 0이 아닌 rating에 대해 평균

        time_list = 0
        for epoch in range(self._epochs):
            t = time.time()
            if epoch ==0:
                print("기존 loss:", mean_squared_error(self._R, np.dot(self._U,self._V.T)))
            
            for i,j,r in self._non_zeros:
                self.gradient_descent(i,j, r)
            time_list += time.time()-t
            rmse , test_rmse= self.calculate_rmse()            
            if (epoch+1) % 10 ==0:
                print("Iteration: %d, train_loss = %.4f, test_loss = %.4f, average time for 1 epoch : %.4f" % (epoch+1, rmse,test_rmse, time_list/10))
                time_list=0

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

    def get_prediction(self, i, j):
        # prediction adding bias
        return self._b + self._b_U[i] + self._b_V[j] + self._U[i, :].dot(self._V[j, :].T)

    def get_complete_matrix(self):
        """
        get complete matrix
        """
        return self._b + self._b_U[:, np.newaxis] + self._b_V[np.newaxis,:] + self._U.dot(self._V.T)

    def gradient_descent(self, i, j, r):
             
        prediction = self.get_prediction(i,j)
        error = r - prediction
        
        self._b_U[i] += self._learning_rate * (error - self._lambda_param * self._b_U[i])
        self._b_V[j] += self._learning_rate * (error - self._lambda_param * self._b_V[j])

        dp, dq = self.gradient(error, i, j)
        self._U[i, :] += self._learning_rate * dp
        self._V[j, :] += self._learning_rate * dq
        
    # 실제 R 행렬과 예측 행렬의 오차를 구하는 함수
    def calculate_rmse(self):
        error = 0

        xi, yi = self._R.nonzero() 
        test_x, test_y = self._R_test.nonzero()

        predicted = self.get_complete_matrix()

        for x, y in zip(xi, yi):
            error += pow(self._R[x, y] - predicted[x, y], 2) 
        
        for i, j in zip(test_x, test_y):
            error += pow(self._R_test[i, j] - predicted[x, y], 2) 

        return np.sqrt(error/len(xi)), np.sqrt(error/len(test_x))