from RFM_model.features.build_features import FeatureBuilder
import requests
import pandas as pd
import os
print(os.getcwd())

raw_path = "data/raw"
SESSIONS_PATH = os.path.join(raw_path, "sessions.jsonl")
USERS_PATH = os.path.join(raw_path, "users.jsonl")
PRODUCTS_PATH = os.path.join(raw_path, "products.jsonl")

sess_df = pd.read_json(path_or_buf=SESSIONS_PATH, lines=True)
usr_df = pd.read_json(path_or_buf=USERS_PATH, lines=True)
prod_df = pd.read_json(path_or_buf=PRODUCTS_PATH, lines=True)
# new_measurement = {
#     "sessions": sess_df.loc[0:3].to_json(),
#     "products": prod_df.loc[0:3].to_json(),
#     "users": usr_df.loc[0:3].to_json(),
# }

new_measurement = {
    "sessions": sess_df.to_json(),
    "products": prod_df.to_json(),
    "users": usr_df.to_json(),
}

response = requests.post('http://127.0.0.1:8000/predict', json=new_measurement)
print(response.content)
