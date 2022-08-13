# API FastAPI du modèle de classification déployée sur Heroku
## Projet 7 du parcours Data Scientist

Cette API est interrogée par un dashboard développé grâce à Streamlit.
Pour interagir avec l'api (tester les requêtes) :

### view when API is launched (index)
https://oc-api-FastAPI-td.herokuapp.com

### answer when asking for score and decision about one customer
https://oc-api-FastAPI-td.herokuapp.com/api/scoring_customer/?SK_ID_CURR=100038

### answer when asking for shap plot params for one selected customer
https://oc-api-FastAPI-td.herokuapp.com/api/shap_plot_params/?SK_ID_CURR=100001

### answer when asking for Cient Information for one selected customer
https://oc-api-FastAPI-td.herokuapp.com/api/client_info/?SK_ID_CURR=100001

### answer when asking for the feature descriptions
https://oc-api-FastAPI-td.herokuapp.com/api/feat_desc/

### answer when asking for a selected feature value for a selected client
https://oc-api-FastAPI-td.herokuapp.com/api/feat_val/?SK_ID_CURR=100001&FEAT_NAME=SK_ID_CURR

### answer when asking for Nearest Neighbors of a selected client samples
https://oc-api-FastAPI-td.herokuapp.com/api/NN_samples/?SK_ID_CURR=100001
