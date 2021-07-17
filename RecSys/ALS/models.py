import numpy as np
import time
import argparse
from sklearn.metrics import mean_squared_error
import dataset

class als_basic():
    def __init__(self, R, factor, lambda_param, epochs):
        """
        :param R: rating Matrix
        :param factor: number of latent factor
        :lambda_param: for regularization
        :epochs: how many update
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._factor = factor
        self._lambda_param = lambda_param
        self._epochs = epochs

    def fit(self):
        #init latent factors
        self._U = np.random.normal(size=(self._num_users, self._factor))
        self._V = np.random.normal(size=(self._num_items, self._factor))
        t = time.time()
        for epoch in range(self._epochs):
            if epoch ==0:
                print("기존 loss:", mean_squared_error(self._R, np.dot(self._U,self._V.T)))
            self.optimize_User_latent()
            self.optimize_Item_latent()
            if epoch % 10 ==0:
                print("Iteration: %d, loss = %.6f, time : %.4f" % (epoch+1, mean_squared_error(self._R, np.dot(self._U,self._V.T)), time.time()-t))

    def optimize_User_latent(self):
        # Fix Item Latent
        V_t = np.transpose(self._V) 
        for n in range(self._num_users):
            V_tV = np.matmul(V_t, self._V)
            lambda_matrix = np.dot(self._lambda_param, np.identity(self._factor))
            inv_matrix = np.linalg.inv(V_tV+ lambda_matrix)
            inv_V_t = np.dot(inv_matrix, V_t)
            result = np.dot(inv_V_t, self._R[n])
            self._U[n] =result
            
    def optimize_Item_latent(self):
        # Fix User Latent
        U_t = np.transpose(self._U)
        for n in range(self._num_items):
            U_tU = np.matmul(U_t, self._U)
            lambda_matrix = np.dot(self._lambda_param, np.identity(self._factor))
            inv_matrix = np.linalg.inv(U_tU+ lambda_matrix)
            inv_U_t = np.dot(inv_matrix, U_t)
            result = np.dot(inv_U_t, self._R.T[n])
            self._V[n] =result
