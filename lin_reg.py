# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:59:48 2020

@author: toztel17
"""
# linear regression function
from scipy import stats
import numpy as np
import pandas as pd
 
class lin_reg:

     def __init__(self):
        self.B = 0 
        self.e = 0 
        self.std_squ = 0
        self.SE = 0
        self.X = 0
        self.CI_upper = []
        self.CI_lower = []
        self.t = 0
        
     def fit(self,y,x):
        if type(x) == str or type(y) == str:
            raise TypeError('Input type cannot be string')
        if len(y[0])>1:
            raise Exception('y cannot have more than 1 column')
#        if isinstance(x,np.ndarray) != True or isinstance(y,np.ndarray) != True:
#            raise TypeError('Input type should be array')
        X = np.hstack([np.ones((len(x),1)),x])
         # listwise deletion
        if pd.DataFrame(X).isnull().values.any() == 'True':
           X = np.array(pd.DataFrame(X).dropna(axis=0,how='any'))
        elif pd.DataFrame(y).isnull().values.any() == 'True':
           y = np.array(pd.DataFrame(y).dropna(axis=0,how='any'))
        self.X = X
        self.B = np.linalg.inv(X.T @ X) @ X.T @ y #b hat
        self.yhat = X @ np.linalg.inv(X.T @ X) @ X.T @ y
        self.e = y-self.yhat
        n = len(x) # row
        k = len(x[0]) # column len(x[0])
        self.std_squ = (self.e.T@self.e)/(n-k-1) #square of the std
        self.SE = np.sqrt(np.diag(np.multiply(self.std_squ,np.linalg.inv(X.T @ X))))
        self.t = stats.t.ppf(.975, n-k-1)
        #95% CI bounds
        for i in range(3):
            self.CI_upper.append(self.B[i] + self.t*self.SE[i]) #upper bound for intercept and slope
            self.CI_lower.append(self.B[i] - self.t*self.SE[i]) # lower bound for intercept and slope