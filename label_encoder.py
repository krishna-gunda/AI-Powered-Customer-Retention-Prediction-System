import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from logging_code import setup_logging
logger=setup_logging('label_encoder')
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


def one_hot_encoder(x_train,x_test):
    try:
        one=OneHotEncoder(drop='first')
        one.fit(x_train[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod']])
        values_train=one.transform(x_train[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod']]).toarray()
        values_test = one.transform(x_test[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod']]).toarray()
        t1=pd.DataFrame(values_train)
        t2=pd.DataFrame(values_test)
        t1.columns=one.get_feature_names_out()
        t2.columns=one.get_feature_names_out()
        x_train.reset_index(drop=True,inplace=True)
        x_test.reset_index(drop=True,inplace=True)
        t1.reset_index(drop=True,inplace=True)
        t2.reset_index(drop=True,inplace=True)
        x_train=pd.concat([x_train,t1],axis=1)
        x_test=pd.concat([x_test,t2],axis=1)
        x_train.drop(columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod'],axis=1)
        x_test.drop(columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod'],axis=1)

        ordinal=OrdinalEncoder()
        ordinal.fit(x_train[[]])

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error_type:{error_type},error_msg:{error_msg},error_line:{error_line}')

