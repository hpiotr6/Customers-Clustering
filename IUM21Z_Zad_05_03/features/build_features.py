#!/usr/bin/env python

from numpy.core.numeric import tensordot
import pandas as pd
from datetime import timedelta
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from logs.logger import log


class FeaturesSimpleModel:
    @log
    def __init__(self, sessions, products, users) -> None:
        self.sess_df = sessions
        self.usr_df = users
        self.prod_df = products

    @classmethod
    @log
    def from_files(cls, raw_path):
        SESSIONS_PATH = os.path.join(raw_path, "sessions.jsonl")
        USERS_PATH = os.path.join(raw_path, "users.jsonl")
        PRODUCTS_PATH = os.path.join(raw_path, "products.jsonl")

        sess_df = pd.read_json(path_or_buf=SESSIONS_PATH, lines=True)
        usr_df = pd.read_json(path_or_buf=USERS_PATH, lines=True)
        prod_df = pd.read_json(path_or_buf=PRODUCTS_PATH, lines=True)

        return cls(sess_df, prod_df, usr_df)

    @classmethod
    @log
    def from_json(cls, sessions, users, products):
        sess_df = pd.read_json(sessions)
        usr_df = pd.read_json(users)
        prod_df = pd.read_json(products)
        return cls(sess_df, prod_df, usr_df)

    def divide_into_train_test_sets(self, frame, test_ratio=0.3):
        train, test = train_test_split(frame, test_size=test_ratio)
        return train, test

    @log
    def divide_data_into_old_and_new(self, sess_df):
        max_date = pd.to_datetime(sess_df["timestamp"].max())
        limit_day = max_date - timedelta(days=30)
        sess_df_old = sess_df.loc[sess_df["timestamp"] < limit_day]
        sess_df_last_month = sess_df.loc[sess_df["timestamp"] >= limit_day]
        sess_df = sess_df_old
        return sess_df_old, sess_df_last_month

    @log
    def create_merge_table(self, old=True):
        sess_old, sess_last_month = self.divide_data_into_old_and_new(
            self.sess_df)
        if old:
            return self.merge_dataframes_and_add_attributes(sess_old)
        return self.merge_dataframes_and_add_attributes(sess_last_month)

    @log
    def merge_dataframes_and_add_attributes(self, sess_df):
        """
        Create Dataframe with many attibutes.It can be helpful when we want to change clustering attributes
        """

        sess_df.product_id = sess_df.product_id.astype(
            pd.Int64Dtype())
        sess_df.dropna(subset=["product_id"], inplace=True)
        merged = sess_df.merge(self.prod_df).merge(self.usr_df)
        customers = (
            merged[["user_id", "price"]]
            .groupby("user_id")
            .sum()
            .rename(columns={"price": "amount"})
            .reset_index()
        )

        ref_date = merged.timestamp.max()
        ref_date = ref_date + timedelta(days=1)
        merged["days_since_last_purchase"] = ref_date - merged.timestamp
        merged["days_since_last_purchase"] = merged["days_since_last_purchase"].astype(
            "timedelta64[D]"
        )
        rec = (
            merged.groupby("user_id")
            .min()
            .reset_index()[["user_id", "days_since_last_purchase"]]
        )

        customers = customers.merge(rec)
        customers.rename(
            columns={"days_since_last_purchase": "recency"}, inplace=True)
        freq = merged[["user_id", "price"]].groupby(
            "user_id").count().reset_index()
        freq.rename(columns={"price": "frequency"}, inplace=True)
        customers = customers.merge(freq)

        self.usr_df["is_male"] = self.usr_df.name.apply(
            lambda name_str: int(name_str.split()[0][-1] != "a")
        )

        customers = customers.merge(self.usr_df[["user_id", "is_male"]])
        merged[["user_id", "event_type"]].groupby("user_id")
        x = merged[["user_id", "event_type"]
                   ].value_counts().reset_index(name="counts")
        viewed = x[x.event_type == "VIEW_PRODUCT"][["counts", "user_id"]]
        bought = x[x.event_type == "BUY_PRODUCT"][["counts", "user_id"]]
        customers = customers.merge(viewed).rename(
            columns={"counts": "viewed_num"})
        customers = customers.merge(bought).rename(
            columns={"counts": "bought_num"})
        customers["bought/sum"] = customers.bought_num / (
            customers.viewed_num + customers.bought_num
        )

        return customers

    @log
    def delete_attributes_except_amount(self, frame):
        frame = frame.drop(
            columns=['is_male', 'viewed_num', 'bought/sum', 'frequency', 'recency', 'bought_num'], axis=1)
        return frame

    @log
    def delete_customers_with_below_10_items_bought(self, frame):
        return frame[frame["bought_num"] >= 10]

    @log
    def generate_processed_files(self, out1_path, minimal=False):

        old_dataframe = self.create_merge_table(old=True)
        new_dataframe = self.create_merge_table(old=False)
        # merged_test = self.merge_dataframes_and_add_attributes()
        # merged_train = self.merge_dataframes_and_add_attributes()

        after_condition_frame_old = self.delete_customers_with_below_10_items_bought(
            old_dataframe)

        # final_frame_train = self.delete_customers_with_below_10_items_bought(merged_train)
        # minimalistic_frame_test = self.delete_attributes_except_amount(final_frame_test)
        if minimal:
            after_condition_frame_new = self.delete_attributes_except_amount(
                new_dataframe)
            after_condition_frame_old = self.delete_attributes_except_amount(
                after_condition_frame_old)

        train_new, test_new = self.divide_into_train_test_sets(
            new_dataframe, test_ratio=0.3)
        train_old, test_old = self.divide_into_train_test_sets(
            after_condition_frame_old, test_ratio=0.3)

        path_test_new = os.path.join(out1_path, "test\\last_month.csv")
        path_train_new = os.path.join(out1_path, "train\\last_month.csv")
        path_test_old = os.path.join(out1_path, "test\\processed.csv")
        path_train_old = os.path.join(out1_path, "train\\processed.csv")
        Path(os.path.join(out1_path, "test")).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(out1_path, "train")).mkdir(
            parents=True, exist_ok=True)

        test_new.to_csv(path_test_new, index=False)
        train_new.to_csv(path_train_new, index=False)
        test_old.to_csv(path_test_old, index=False)
        train_old.to_csv(path_train_old, index=False)

    # def generate_processed_files_maximal(self, out_path):
    #     path = os.path.join(out_path, "processed.csv")
    #     merged = self.merge_dataframes_and_add_attributes()
    #     final_frame = self.delete_customers_with_below_10_items_bought(merged)
    #     final_frame.to_csv(path, index=False)


if __name__ == '__main__':
    model = FeaturesSimpleModel.from_files(
        "E:/code/gitlab elka repo/ium-21z/data/raw")
    model.generate_processed_files(
        "E:/code/gitlab elka repo/ium-21z/data/processed")
