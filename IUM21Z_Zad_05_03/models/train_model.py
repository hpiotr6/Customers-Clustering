# imports
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from kneed import KneeLocator
import os
import pickle


class SimpleModelTrainer:
    def __init__(self, filepath, attribute='amount') -> None:
        self.df = self.create_df(filepath)
        self.attribute = attribute

    def get_cluster_number(self):
        Sum_of_squared_distances = []

        K = range(1, 11)

        for num_clusters in K:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(self.df[[self.attribute]])
            Sum_of_squared_distances.append(kmeans.inertia_)

        kl = KneeLocator(
            range(1, 11), Sum_of_squared_distances, curve="convex", direction="decreasing"
        )
        return kl.elbow

    def train_simple_model(self):
        k = self.get_cluster_number()

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.df[self.attribute])
        return kmeans

    def train_save_simple_model(self, output_filepath):
        trained_model = self.train_simple_model()
        path = os.path.join(output_filepath, "simple_model.pkl")
        pickle.dump(trained_model, open(path, 'wb'))  # .pkl
