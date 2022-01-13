from numpy import mod
from pandas.core.frame import DataFrame
from pandas.core.reshape.merge import merge
from IUM21Z_Zad_05_03.features.build_features import FeaturesSimpleModel
from IUM21Z_Zad_05_03.models.train_model import SimpleModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class TestFeatures:
    def test_from_files(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        assert type(model) == FeaturesSimpleModel

    def test_from_json(self):

        sessions = "data/raw/sessions.jsonl"
        users = "data/raw/users.jsonl"
        products = "data/raw/products.jsonl"

        model = FeaturesSimpleModel.from_json(
            sessions, users, products)
        assert type(model) == FeaturesSimpleModel

    def test_merge_dataframes_and_add_attributes_type(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        assert type(frame) == pd.DataFrame

    def test_merge_dataframes_and_add_attributes_size(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        assert frame.size > 400

    def test_merge_dataframes_and_add_attributes_check_if_attributes_exist(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        for column_name in ['is_male', 'amount', 'recency', 'frequency', 'viewed_num', 'bought_num', 'bought/sum']:
            if column_name in frame.columns:
                assert True
            else:
                assert False

    def test_delete_attributes_except_amount(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        minimal_frame = model.delete_attributes_except_amount(frame)
        # for column_name in ['recency']:
        #     assert column_name not in minimal_frame.columns
        print(minimal_frame.columns)
        assert 'is_male' not in minimal_frame.columns
        assert 'bought_num' not in minimal_frame.columns

        assert 'amount' in minimal_frame.columns
        assert len(minimal_frame.columns) == 2

    def test_delete_customers_with_below_10_items_bought(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        frame = model.merge_dataframes_and_add_attributes()
        frame = model.delete_customers_with_below_10_items_bought(frame)
        assert frame['viewed_num'].min() >= 10

    def test_generate_processed_files_minimal(self):
        model = FeaturesSimpleModel.from_files("data/raw")

        model.generate_processed_files(
            "E:\code\gitlab elka repo\ium-21z\data\processed")
        assert os.path.isfile("data/processed/train/processed.csv")
        s1 = SimpleModelTrainer("data/processed/train/processed.csv")
        assert len(s1.df.columns) == 2

    def test_generate_processed_files_maximal(self):
        model = FeaturesSimpleModel.from_files("data/raw")
        model.generate_processed_files_maximal(
            "E:\code\gitlab elka repo\ium-21z\data\processed")

        assert os.path.isfile("data/processed/processed.csv")
        s1 = SimpleModelTrainer("data/processed/processed.csv")
        assert len(s1.df.columns) == 8
