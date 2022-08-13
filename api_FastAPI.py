# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
import sklearn
import json
from sklearn.neighbors import NearestNeighbors

import shap

# File system manangement
import os
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import pickle

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

app = FastAPI()

#######################################################################################################
# Load DATA
#######################################################################################################
print('*************    Loading DATA ...')
# Train data
application_train_domain = pd.read_csv('./data/application_train_domain.csv')
#print(application_train_domain.shape)

# Test data
application_test_domain = pd.read_csv('./data/application_test_domain.csv')

# Transformed Train data
df_X_train = pd.read_csv('./data/df_X_train.csv')

# transformed Test data
df_X_test = pd.read_csv('./data/df_X_test.csv')

# Test 1000 samples data
df_X_test_1000 = pd.read_csv('./data/df_X_test_1000.csv')

# Test 1000 samples data
df_feat_desc = pd.read_csv('./data/HomeCredit_columns_description.csv')

# Cient Information data
df_Client_Info = pd.read_csv('./data/df_Client_Info.csv')

# test neighbors data (50)
df_test_neighbors = pd.read_csv('./data/df_test_neighbors.csv')


### SHAP   ### 
# expected value
with open("./shap/expected_value.pkl", "rb") as f:
    expected_value = pickle.load(f)
    print(expected_value)
# shap values
with open("./shap/shap_values.pkl", "rb") as f:
    shap_values = pickle.load(f)


# Loading the trained model
with open("./model/best_model_optim_threshold.pkl", "rb") as f:
    best_model, optimal_threshold = pickle.load(f)
    print(optimal_threshold)
    print(best_model.get_params(deep=True))


#######################################################################################################
# Requetes GET
#######################################################################################################

# answer when asking for sk_ids
# Test local: http://127.0.0.1:8000/api/sk_ids/
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/sk_ids/
@app.get("/api/sk_ids/")
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the df_X_test_1000 dataframe
    sk_ids = pd.Series(list(application_test_domain['SK_ID_CURR'][:1000]))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return {'status': 'ok',
            'data': sk_ids_json}


# answer when asking for score and decision about one customer
# Test local : http://127.0.0.1:8000/api/scoring_customer/?SK_ID_CURR=100038
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/scoring_customer/?SK_ID_CURR=100038
@app.get("/api/scoring_customer/")
def scoring_customer(SK_ID_CURR : int = 100038):
    print('SK_ID_CURR :',SK_ID_CURR)
    #compute the index of SK_ID_CURR
    ind = application_test_domain[application_test_domain['SK_ID_CURR']==SK_ID_CURR].index
    # compute the score
    score = best_model.predict_proba(df_X_test_1000.iloc[ind,:])[0][1] # Default proba 
    print('score :',score)
    return {"score": score, "optimal_threshold": optimal_threshold}


# answer when asking for shap plot params for one selected customer
# Test local : http://127.0.0.1:8000/api/shap_plot_params/?SK_ID_CURR=100001
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/shap_plot_params/?SK_ID_CURR=100001
@app.get("/api/shap_plot_params/")
def shap_plot_params(SK_ID_CURR : int = 100001):
    print('SK_ID_CURR :',SK_ID_CURR)
    #compute the index of SK_ID_CURR
    ind = application_test_domain[application_test_domain['SK_ID_CURR']==SK_ID_CURR].index
    # compute the shap plot params
    expected_value_1 = expected_value[1]
    shap_values_1 = pd.DataFrame(shap_values[1][ind,:])
    selected_sample = df_X_test_1000.loc[ind]
    # Convert the pd.Series (df row) of customer's data to JSON
    shap_values_1_json = json.loads(shap_values_1.to_json())
    selected_sample_json = json.loads(selected_sample.to_json())
    # Return the data
    return {
            "expected_value_1": expected_value_1, 
            "shap_values_1": shap_values_1_json, 
            "selected_sample": selected_sample_json
           }


# answer when asking for Cient Information for one selected customer
# Test local : http://127.0.0.1:8000/api/client_info/?SK_ID_CURR=100001
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/client_info/?SK_ID_CURR=100001
@app.get("/api/client_info/")
def client_info(SK_ID_CURR : int = 100001):
    print('SK_ID_CURR :',SK_ID_CURR)
    #compute the index of SK_ID_CURR
    ind = df_Client_Info[df_Client_Info['SK_ID_CURR']==SK_ID_CURR].index[0]
    
    # load the client info
    info_cols = ["SK_ID_CURR",
                 "YEARS_BIRTH", 
                 "YEARS_EMPLOYED",
                 "NAME_INCOME_TYPE",
                 "AMT_INCOME_TOTAL",
                 "NAME_CONTRACT_TYPE",
                 "NAME_FAMILY_STATUS",
                 "GENDER",
                 "NAME_EDUCATION_TYPE"]
    selected_client_info = df_Client_Info[info_cols].loc[ind]
    # Convert the pd.Series (df row) of customer's data to JSON
    selected_client_info_json = json.loads(selected_client_info.to_json())
    # Return the data
    return {"selected_client_info": selected_client_info_json
           }

### answer when asking for a selected feature value for a selected client
# Test local : http://127.0.0.1:8000/api/feat_desc/
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/feat_desc/
@app.get("/api/feat_desc/")
def feat_desc():
    # Convert the pd.Series (df row) of customer's data to JSON
    feat_desc_json = json.loads(df_feat_desc.to_json())
    # Return the data
    return {
            "feat_desc": feat_desc_json
           }

# answer when asking for the feature descriptions
# Test local : http://127.0.0.1:8000/api/feat_val/?SK_ID_CURR=100001&FEAT_NAME=SK_ID_CURR
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/feat_val/?SK_ID_CURR=100001&FEAT_NAME=SK_ID_CURR
@app.get("/api/feat_val/")
def feat_val(SK_ID_CURR : int = 100001, FEAT_NAME : str = 'SK_ID_CURR'):
    print('SK_ID_CURR :',SK_ID_CURR)
    print('FEAT_NAME :',FEAT_NAME)
    #compute the index of SK_ID_CURR
    ind = application_test_domain[application_test_domain['SK_ID_CURR']==SK_ID_CURR].index
    feat_val = list(application_test_domain.loc[ind, FEAT_NAME])
    # Return the data
    return {
            "feat_val": feat_val
           }

### answer when asking for Nearest Neighbors of a selected client samples
# Test local : http://127.0.0.1:8000/api/NN_samples/?SK_ID_CURR=100001
# Test Heroku : https://oc-api-FastAPI-td.herokuapp.com/api/NN_samples/?SK_ID_CURR=100001
@app.get("/api/NN_samples/")
def NN_samples(SK_ID_CURR : int = 100001):
    print('SK_ID_CURR :',SK_ID_CURR)
    #compute the index of SK_ID_CURR
    ind = application_test_domain[application_test_domain['SK_ID_CURR']==SK_ID_CURR].index
    neighbors_indexes = df_test_neighbors.loc[ind,:].values[0]
    stat_cols = ["YEARS_BIRTH", "YEARS_EMPLOYED", "NAME_INCOME_TYPE", "AMT_INCOME_TOTAL", "NAME_CONTRACT_TYPE", \
                 "NAME_FAMILY_STATUS", "GENDER", "NAME_EDUCATION_TYPE"]
    NN_samples = pd.DataFrame(columns=stat_cols) 
    for i in range(len(neighbors_indexes)):
        ind_samp = neighbors_indexes[i]
        NN_samples.loc[i] = df_Client_Info[stat_cols].loc[ind_samp]
    
    # Convert the pd.Series (df row) of customer's data to JSON
    NN_samples_json = json.loads(NN_samples.to_json())
    # Return the data
    return {
            "NN_samples": NN_samples_json
           }

# homepage route
@app.get("/")
def read_root():
    return {"message": "FastAPI Home Page...OC_Projet7_FastAPI_TD"}
