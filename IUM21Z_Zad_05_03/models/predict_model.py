import pickle
import os


class SimpleModelPredictor:
    def __init__(self, model_name) -> None:
        self.model = self.load(model_name)

    def load_model(self, filepath):
        path = os.path.join(filepath, "my_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    def predict_simple_model(self, frame):
        y_kmeans = self.model.predict(frame)
        clustered = frame.assign(Cluster=y_kmeans)
        result = clustered[clustered.Cluster == 0].user_id
        return result
