import pickle
import os
from urllib.parse import uses_relative
from logs.logger import log
import numpy as np


class SimpleModelPredictor:
    @log
    def __init__(self, path) -> None:
        self.model = self.load_model(path)

    @log
    def load_model(self, filepath):
        path = os.path.join(filepath, "simple_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    @log
    def predict_simple_model(self, frame):
        y_kmeans = self.model.predict(frame[["amount"]])
        clustered = frame.assign(Cluster=y_kmeans)
        return clustered

    @log
    def predict_simple_model_from_path(self, frame):
        y_kmeans = self.model.predict(frame[["amount"]])
        clustered = frame.assign(Cluster=y_kmeans)
        return clustered

    @log
    def best_group(self, clustered_frame):
        values = list(set(clustered_frame["Cluster"].values))
        max_mean_spending = 0
        best_group = -1
        for value in values:
            mean_spendings = clustered_frame.loc[clustered_frame['Cluster'] == value]["amount"].mean(
            )
            if mean_spendings > max_mean_spending:
                best_group = value
                max_mean_spending = mean_spendings
        return best_group

    def get_mean_spendings_of_group(self, cluster, df_old, df_new=None):
        user_ids_np = df_old["user_id"].to_numpy()
        unique_arr = np.unique(user_ids_np)
        user_ids = list(set(unique_arr))

        if df_new is not None:
            df_correct_cluster = df_old.loc[df_old['Cluster'] == cluster]
            user_ids_np = df_correct_cluster["user_id"].to_numpy()
            unique_arr = np.unique(user_ids_np)
            user_ids_correct_cluster = list(set(unique_arr))
            selected_new_df = df_new.loc[df_new["user_id"].isin(user_ids)]
            df_new = selected_new_df.loc[selected_new_df["user_id"].isin(
                user_ids_correct_cluster)]

            return df_new["amount"].mean()
        # return df_old.loc[df_old["user_id"]]
    # def get_averages_spendings_in_last_month():
