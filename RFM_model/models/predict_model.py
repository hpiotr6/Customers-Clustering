import pickle
import os
import numpy as np


class Predictor:
    def __init__(self, filepath) -> None:
        self.model = self.load_model(filepath)

    def load_model(self, filepath):
        path = os.path.join(filepath, "my_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    def predict(self, X):
        labels = self.model.predict(X)
        clustered = X.assign(Cluster=labels)
        summary = clustered.groupby(['Cluster']).agg({'Recency': 'mean',
                                                      'Frequency': 'mean',
                                                      'MonetaryValue': ['mean', 'count'], }).round(0)
        which_cluster = np.argmax(summary.MonetaryValue["mean"])
        result = clustered[clustered.Cluster == which_cluster].user_id
        return result
