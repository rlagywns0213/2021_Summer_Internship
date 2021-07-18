import numpy as np
import time
import argparse
from sklearn.metrics import mean_squared_error
import dataset

class occf_bottleneck():
    def __init__(self, R, R_test, factor, lambda_param, epochs):
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
        self._lambda_param = lambda_param
        self._epochs = epochs
        self._P = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(R), dtype = np.float64) #binary preference
        self._alpha = 40
        self._C = 1 + self._alpha * self._R # Confidence Matrix size (m, n)

    def fit(self):
        #init latent factors
        np.random.seed(0)
        self._U = np.random.normal(size=(self._num_users, self._factor))
        self._V = np.random.normal(size=(self._num_items, self._factor))
        # self._U = np.random.normal(scale = 1.0/self._factor, size=(self._num_users, self._factor))
        # self._V = np.random.normal(scale = 1.0/self._factor, size=(self._num_items, self._factor))
        
        for epoch in range(self._epochs):
            t = time.time()
            if epoch ==0:
                print("기존 loss:", mean_squared_error(self._R, np.dot(self._U,self._V.T)))
                # for bottleneck problem
            self._V_t = np.transpose(self._V) 
            self._V_t_V = self._V_t.dot(self._V)
            self.optimize_User_latent()

            self._U_t = np.transpose(self._U)
            self._U_t_U = self._U_t.dot(self._U)
            self.optimize_Item_latent()
            start_rank = time.time()
            rank= self.compute_rank()

            print("time to compute rank : %.4f" % ( time.time() - start_rank ))
            print("Iteration: %d, loss = %.6f, test_rank = %.4f, time : %.4f" % (epoch+1, mean_squared_error(self._R, np.dot(self._U,self._V.T)),rank, start_rank-t))

    def optimize_User_latent(self):
        # Fix Item Latent
        for n in range(self._num_users):
            C_u = np.diag(self._C[n, :])
            # V_t_C_u_V = V_t.dot(C_u).dot(self._V)
            reduce_bottlenec= self._V_t_V + self._V_t.dot(C_u - np.identity(self._num_items)).dot(self._V)
            lambda_matrix = self._lambda_param* np.identity(self._factor)
            before_inv_matrix = reduce_bottlenec+lambda_matrix
            inv_matrix = np.linalg.inv(before_inv_matrix)   
            inv_V_t = np.dot(inv_matrix, self._V_t)
            result = inv_V_t.dot(C_u).dot(self._P[n])
            self._U[n,:] =result
            
    def optimize_Item_latent(self):
        # Fix User Latent
        for i in range(self._num_items):
            C_u = np.diag(self._C[:,i])
            # U_t_C_u_U = U_t.dot(C_u).dot(self._U)
            reduce_bottlenec = self._U_t_U + self._U_t.dot(C_u - np.identity(self._num_users)).dot(self._U)
            lambda_matrix = np.dot(self._lambda_param, np.identity(self._factor))
            before_inv_matrix = reduce_bottlenec+lambda_matrix
            inv_matrix = np.linalg.inv(before_inv_matrix)
            inv_U_t = np.dot(inv_matrix, self._U_t)
            result = inv_U_t.dot(C_u).dot(self._P[:,i])
            self._V[i, :] =result

    def compute_rank(self):
        prediction = self._U.dot(self._V.T)
        temp_1 = 0
        temp_2 = 0
        
        for x in range(self._num_users):
            inv_pre = -1 * prediction[x, :]
            sort_x = inv_pre.argsort() # index starts with 0
            # sort_x = sort_x.argsort()
            rank_x = sort_x / len(sort_x)
            
            temp_1 += (self._R_test[x, :] * rank_x).sum()
            temp_2 += self._R_test[x, :].sum()
        
        rank = temp_1 / temp_2
            
        return rank