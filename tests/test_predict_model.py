from IUM21Z_Zad_05_03.models.predict_model import SimpleModelPredictor
from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer
from sklearn.cluster import KMeans


class TestSimpleModelPredictor:
    def test_model_create(self):
        model = SimpleModelPredictor("models")
        assert type(model.model) == KMeans

    def test_predict(self):
        model = SimpleModelPredictor("models")

        # to get frame
        trainer = SimpleModelTrainer("data/processed/processed.csv")

        result_frame = model.predict_simple_model(trainer.df)

        # k is 3, to it will be 3 clusters
        assert list(set(result_frame["Cluster"].values)) == [0, 1, 2]

    def test_best_group(self):
        model = SimpleModelPredictor("models")

        # to get frame
        trainer = SimpleModelTrainer("data/processed/processed.csv")

        result_frame = model.predict_simple_model(trainer.df)
        best_group = model.best_group(result_frame)
        max_spendings = result_frame.loc[result_frame['Cluster']
                                         == best_group]["amount"].mean()
        # clusters number
        for i in range(3):
            if i != best_group:
                mean_spendings = result_frame.loc[result_frame['Cluster'] == i]["amount"].mean(
                )
                assert mean_spendings <= max_spendings

    def test_simple_model_criterium(self):
        """Not Completed Test
        """
        model = SimpleModelPredictor("models")

        result_frame = model.predict_simple_model(trainer.df)
        best_group = model.best_group(result_frame)
