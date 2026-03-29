import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
import logging
from logging_code import setup_logging
logger=setup_logging('main')
from sklearn.model_selection import train_test_split
from missing_data import handling_data


class CHURN():
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'total data shape: {self.df.shape}')
            logger.info(f'total column names before adding: {self.df.columns}')
            payment_to_sim = {
                'Electronic check': 'Reliancejio',
                'Mailed check': 'Airtel',
                'Bank transfer (automatic)': 'BSNL',
                'Credit card (automatic)': 'Idea'
            }

            self.df['sim_column'] = self.df['PaymentMethod'].map(payment_to_sim)
            logger.info("---------------------------------------------------------------------")
            logger.info(f'total column names after adding: {self.df.columns}')
            logger.info("-"*100)
            logger.info("dividing the data into dependent and independent columns")
            self.y = self.df['Churn']
            self.x=self.df.drop("Churn",axis=1)

            logger.info(f'independent columns:{self.x.columns}')
            logger.info(f'dependent columns:{self.y.name}') # since it is single column we use name
            #changing the dependent column into numerical
            self.y=self.y.map({'Yes': 1, 'No': 0}).astype(int)

            logger.info ('dividing the data into x_train,x_test,y_train,y_test')
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(
                f'x_train:{len(self.x_train)}, y_train:{len(self.y_train)}, x_test:{len(self.x_test)}, y_test:{len(self.y_test)}')

            logger.info(f'null values in x_train:{self.x_train['TotalCharges'].isnull().sum()
            }null values in x_test:{self.x_test['TotalCharges'].isnull().sum()}')
            #print(f'{self.x_train.info()}\n')
            #print(f'{self.x_test.info()}')
            #from above operations we can to know that the TotalCharges in the table in the string format we nne to chnage into numerical
            self.x_train['TotalCharges']=self.x_train['TotalCharges'].replace(' ',np.nan)
            self.x_train['TotalCharges']=pd.to_numeric(self.x_train['TotalCharges'])
           # self.x_train.info()
            self.x_test['TotalCharges'] = self.x_test['TotalCharges'].replace(' ', np.nan)
            self.x_test['TotalCharges'] = pd.to_numeric(self.x_test['TotalCharges'])
           # self.x_test.info()
        # we changed the totalcharges column into numerical
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error_type:{error_type},error_msg:{error_msg},error_line:{error_line}')

    def handling_missing_data(self):
        try:
            logger.info(f'Before handling the null values')
            logger.info(f'the no of null values in the x_train[TotalCharges]:{self.x_train['TotalCharges'].isnull().sum()}')
            logger.info(f'the no of null values in the x_test[totalcharges]:{self.x_test['TotalCharges'].isnull().sum()}')
            self.x_train,self.x_test=handling_data(self.x_train,self.x_test)
            logger.info(f'After handling the null values')
            logger.info(
                f'the no of null values in the x_train[TotalCharges]:{self.x_train['TotalCharges'].isnull().sum()}')
            logger.info(
                f'the no of null values in the x_test[totalcharges]:{self.x_test['TotalCharges'].isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error_type:{error_type},error_msg:{error_msg},error_line:{error_line}')





if __name__ == '__main__':
    try:
        obj=CHURN('WA_Fn-UseC_-Telco-Customer-Churn (1).csv')
        obj.handling_missing_data()
    except Exception as e:
        err_type,err_msg,err_line=sys.exc_info()
        logger.info(f'error_type:{err_type},error_msg:{err_msg},error_line:{err_line}')
