import numpy as np
import time
from sklearn.metrics import mean_squared_error

class Probabilistic_MF():
    def __init__(self, R, R_test, factor, lambda_u, lambda_v, learning_rate, epochs):
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
        self._lambda_U = lambda_u
        self._lambda_V = lambda_v
        self._epochs = epochs
        self._I = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(R), dtype = np.float64) #Indicator (nonzero쓰면 되지 않는가)
    
    def fit(self):
        #init latent factors
        np.random.seed(0)
        self._U = np.random.normal(0,0.1,size = (self._num_users,self._factor))
        self._V = np.random.normal(0,0.1,size = (self._num_items,self._factor))

        time_list = 0
        for epoch in range(self._epochs):
            t = time.time()
            
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0 :
                        self.gradient_descent(i, j, self._R[i, j])
            time_list += time.time()-t
            rmse , test_rmse= self.calculate_rmse()             
            if (epoch+1) % 10 ==0:
                print("Iteration: %d, train_loss = %.4f, test_loss = %.4f, average time for 1 epoch : %.4f" % (epoch+1, rmse,test_rmse, time_list/10))
                time_list=0

    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD
        param error : rating - prediction error
        param u : user index
        param i : item index
        """
        
        dp = (error * self._V[j, :]) - (self._lambda_U * self._U[i, :])
        dq = (error * self._U[i, :]) - (self._lambda_V * self._V[j, :])
    
        return dp, dq

    def gradient_descent(self, i, j, R):
             
        prediction = np.dot(self._U[i,:], self._V[j,:].T)
        error = R - prediction
        
        dp, dq = self.gradient(error, i, j)
        self._U[i, :] += self._learning_rate * dp
        self._V[j, :] += self._learning_rate * dq
        
    # 실제 R 행렬과 예측 행렬의 오차를 구하는 함수
    def calculate_rmse(self):
        train_loss = 0
        test_loss = 0
        xi, yi = self._R.nonzero() 
        test_x, test_y = self._R_test.nonzero()

        predicted = self._U.dot(self._V.T)

        for x, y in zip(xi, yi):
            train_loss += pow(self._R[x, y] - predicted[x, y], 2) 
        
        for i, j in zip(test_x, test_y):
            test_loss += pow(self._R_test[i, j] - predicted[x, y], 2) 

        return np.sqrt(train_loss/len(xi)), np.sqrt(test_loss/len(test_x))