from sklearn.cluster import KMeans
import pandas as pd
import os
import pickle
from logs.logger import log


class Trainer:
    @log
    def __init__(self, n_clusters, random_state, filepath) -> None:
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.df = self.create_df(filepath)

    @log
    def create_df(self, filepath):
        return pd.read_csv(filepath)

    @log
    def train(self):
        return self.kmeans.fit(self.df)

    @log
    def train_save(self, output_filepath):
        trained_model = self.train()
        path = os.path.join(output_filepath, "my_model.pkl")
        pickle.dump(trained_model, open(path, 'wb'))  # .pkl
