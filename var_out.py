import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger=setup_logging('var_out')
import sys

def variable_outliers(x_train,x_test):
    try:
        logger.info(f'before column names:{x_train.columns}')
        logger.info(f'before column names:{x_test.columns}')
        x_train['TotalCharges_var']=np.sqrt(x_train['TotalCharges'])
        x_test['TotalCharges_var']=np.sqrt(x_test['TotalCharges'])
        x_train=x_train.drop(['TotalCharges'],axis=1)
        x_test=x_test.drop(['TotalCharges'],axis=1)
        logger.info(f'after column names:{x_train.columns}')
        logger.info(f'after column names:{x_test.columns}')
        return x_train,x_test
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error type{error_type},error msg {error_msg} ,error_line {error_line}')