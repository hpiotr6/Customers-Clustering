import pickle
import os


class SimpleModelPredictor:
    def __init__(self, path) -> None:
        self.model = self.load_model(path)

    def load_model(self, filepath):
        path = os.path.join(filepath, "simple_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    def predict_simple_model(self, frame):
        y_kmeans = self.model.predict(frame[["amount"]])
        clustered = frame.assign(Cluster=y_kmeans)
        return clustered

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
