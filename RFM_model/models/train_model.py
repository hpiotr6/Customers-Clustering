from sklearn.cluster import KMeans
import pandas as pd
import os
import pickle


class Trainer:
    def __init__(self, n_clusters, random_state, filename) -> None:
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.df = self.create_df(filename)

    def create_df(self, filename):
        path = os.path.join("../../data/processed", filename)
        return pd.read_csv(path)

    def train(self):
        return self.kmeans.fit(self.df)

    def train_save(self, output_filename):
        trained_model = self.train()
        path = os.path.join("../../models", output_filename)  # extension .pkl
        pickle.dump(trained_model, open(path, 'wb'))
