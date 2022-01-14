import pickle
import os
import numpy as np
from logs.logger import log


class Predictor:
    @log
    def __init__(self, filepath) -> None:
        self.model = self.load_model(filepath)

    @log
    def load_model(self, filepath):
        path = os.path.join(filepath, "my_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    @log
    def predict(self, X):
        labels = self.model.predict(X)
        clustered = X.assign(Cluster=labels)
        summary = clustered.groupby(['Cluster']).agg({'Recency': 'mean',
                                                      'Frequency': 'mean',
                                                      'MonetaryValue': ['mean', 'count'], }).round(0)
        which_cluster = np.argmax(summary.MonetaryValue["mean"])
        result = clustered[clustered.Cluster == which_cluster].user_id
        return result

    @log
    def predict_and_return_best_cluster(self, X):
        labels = self.model.predict(X)
        clustered = X.assign(Cluster=labels)
        summary = clustered.groupby(['Cluster']).agg({'Recency': 'mean',
                                                      'Frequency': 'mean',
                                                      'MonetaryValue': ['mean', 'count'], }).round(0)
        which_cluster = np.argmax(summary.MonetaryValue["mean"])
        return which_cluster

    @log
    def predict_and_return_prediccted_df(self, X):
        labels = self.model.predict(X)
        clustered = X.assign(Cluster=labels)

        return clustered

    @log
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
