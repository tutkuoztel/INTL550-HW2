# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:53:47 2020

@author: toztel17
"""

#TEST Code

import unittest
from lin_reg import lin_reg
import numpy as np

class lin_reg_test(unittest.TestCase):
    def test_fit(self):
        x = 'tutku'
        y = 'oztel'
        r = lin_reg()
        with self.assertRaises(TypeError): r.fit(x,y)
        
        
        y = [1,2,3,4]
        x = [5,6,7,8]
        x = np.array(x)
        y = np.array(y).reshape(2,2)
    
        with self.assertRaises(Exception): r.fit(y,x)

if __name__ == '__main__':
    unittest.main()