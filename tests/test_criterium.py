from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer
from sklearn.model_selection import train_test_split
from IUM21Z_Zad_05_03.models.predict_model import SimpleModelPredictor
from IUM21Z_Zad_05_03.features.build_features import FeaturesSimpleModel
from RFM_model.models.predict_model import Predictor


def test_criterium_simple_model():
    test_old_df = SimpleModelTrainer(
        filepath="data/processed/test/processed.csv").df
    test_new_df = SimpleModelTrainer(
        filepath="data/processed/test/last_month.csv").df
    predictor = SimpleModelPredictor("models")
    predicter_frame = predictor.predict_simple_model(test_old_df)
    best_group = predictor.best_group(predicter_frame)

    max = predictor.get_mean_spendings_of_group(
        best_group, predicter_frame, test_new_df)

    # test if mean values of other groups are lower - analythical condition
    for i in range(3):
        if i != best_group:
            pred = predictor.get_mean_spendings_of_group(
                i, predicter_frame, test_new_df)
            assert max > pred


def test_criterium_complicated_model():
    test_old_df = SimpleModelTrainer(
        filepath="data/processed/test/processed.csv").df
    test_new_df = SimpleModelTrainer(
        filepath="data/processed/test/last_month.csv").df
    predictor = Predictor("models")

    predicter_frame = predictor.predict(test_old_df)
    best_group = predictor.best_group(predicter_frame)

    max = predictor.get_mean_spendings_of_group(
        best_group, predicter_frame, test_new_df)
    for i in range(3):
        if i != best_group:
            pred = predictor.get_mean_spendings_of_group(
                i, predicter_frame, test_new_df)
            assert max > pred
