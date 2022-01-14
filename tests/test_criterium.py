from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer
from sklearn.model_selection import train_test_split
from IUM21Z_Zad_05_03.models.predict_model import SimpleModelPredictor
from IUM21Z_Zad_05_03.features.build_features import FeaturesSimpleModel
from RFM_model.models.predict_model import Predictor
from RFM_model.features.build_features import FeatureBuilder
import pandas as pd


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
    # predicter_frame = predictor.predict(
    #     test_old_df["Cluster"]["Recency"]["Frequency"]["MonetaryValue"])
    x = FeatureBuilder.from_files("data/raw").build()
    df_with_predicts = predictor.predict_and_return_prediccted_df(x)
    best_group = predictor.predict_and_return_best_cluster(x)
    maximum = predictor.get_mean_spendings_of_group(
        best_group, df_with_predicts, test_new_df)
    for i in range(3):
        if i != best_group:
            pred = predictor.get_mean_spendings_of_group(
                i, df_with_predicts, test_new_df)
            assert maximum > pred


def test_criterium_simple_model_all_data():
    test_old_df1 = SimpleModelTrainer(
        filepath="data/processed/test/processed.csv").df
    test_old_df2 = SimpleModelTrainer(
        filepath="data/processed/train/processed.csv").df
    test_old_df = pd.concat([test_old_df1, test_old_df2])
    test_new_df1 = SimpleModelTrainer(
        filepath="data/processed/test/last_month.csv").df
    test_new_df2 = SimpleModelTrainer(
        filepath="data/processed/train/last_month.csv").df
    test_new_df = pd.concat([test_new_df1, test_new_df2])

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


def test_criterium_complicated_model_all_data():
    test_old_df1 = SimpleModelTrainer(
        filepath="data/processed/test/processed.csv").df
    test_old_df2 = SimpleModelTrainer(
        filepath="data/processed/train/processed.csv").df
    test_old_df = pd.concat([test_old_df1, test_old_df2])
    test_new_df1 = SimpleModelTrainer(
        filepath="data/processed/test/last_month.csv").df
    test_new_df2 = SimpleModelTrainer(
        filepath="data/processed/train/last_month.csv").df
    test_new_df = pd.concat([test_new_df1, test_new_df2])
    predictor = Predictor("models")
    # predicter_frame = predictor.predict(
    #     test_old_df["Cluster"]["Recency"]["Frequency"]["MonetaryValue"])
    x = FeatureBuilder.from_files("data/raw").build()
    df_with_predicts = predictor.predict_and_return_prediccted_df(x)
    best_group = predictor.predict_and_return_best_cluster(x)
    maximum = predictor.get_mean_spendings_of_group(
        best_group, df_with_predicts, test_new_df)
    for i in range(3):
        if i != best_group:
            pred = predictor.get_mean_spendings_of_group(
                i, df_with_predicts, test_new_df)
            assert maximum > pred
