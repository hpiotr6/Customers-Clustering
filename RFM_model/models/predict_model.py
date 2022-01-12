import pickle
import os


class Predictor:
    def __init__(self, model_name) -> None:
        self.model = self.load_model(model_name)

    def load_model(self, filepath):
        path = os.path.join(filepath, "my_model.pkl")
        loaded_model = pickle.load(open(path, 'rb'))
        return loaded_model

    def predict(self, X):
        labels = self.model.predict(X)
        result = X.assign(Cluster=labels)
        return result
