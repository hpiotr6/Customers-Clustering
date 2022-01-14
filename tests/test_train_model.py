from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer
from sklearn.cluster import KMeans
import os


class TestSimpleModelTrainer:

    def test_create_instance(self):
        s1 = SimpleModelTrainer(filepath="data/processed/train/processed.csv")
        assert type(s1) == SimpleModelTrainer

    def test_get_clusters_number(self):
        s1 = SimpleModelTrainer(filepath="data/processed/train/processed.csv")
        clusters_number = s1.get_clusters_number()
        assert clusters_number == 3

    def test_train_simple_model_type(self):
        s1 = SimpleModelTrainer(filepath="data/processed/train/processed.csv")
        model = s1.train_simple_model()
        assert type(model) == KMeans

    def test_train_save_model_exists(self):
        s1 = SimpleModelTrainer(filepath="data/processed/train/processed.csv")
        s1.train_save_simple_model("models")
        assert os.path.isfile("models/simple_model.pkl")
