import pandas as pd
from datetime import timedelta
import os


class FeaturesSimpleModel:
    def __init__(self, sessions, products, users) -> None:
        self.sess_df = sessions
        self.usr_df = users
        self.prod_df = products

    @classmethod
    def from_files(cls, raw_path):
        SESSIONS_PATH = os.path.join(raw_path, "sessions.jsonl")
        USERS_PATH = os.path.join(raw_path, "users.jsonl")
        PRODUCTS_PATH = os.path.join(raw_path, "products.jsonl")

        sess_df = pd.read_json(path_or_buf=SESSIONS_PATH, lines=True)
        usr_df = pd.read_json(path_or_buf=USERS_PATH, lines=True)
        prod_df = pd.read_json(path_or_buf=PRODUCTS_PATH, lines=True)

        return cls(sess_df, prod_df, usr_df)

    @classmethod
    def from_json(cls, sessions, users, products):
        sess_df = pd.read_json(sessions)
        usr_df = pd.read_json(users)
        prod_df = pd.read_json(products)
        return cls(sess_df, prod_df, usr_df)

    def merge_dataframes_and_add_attributes(self):
        """
        Create Dataframe with many attibutes.It can be helpful when we want to change clustering attributes
        """
        self.sess_df.product_id = self.sess_df.product_id.astype(
            pd.Int64Dtype())
        self.sess_df.dropna(subset=["product_id"], inplace=True)
        merged = self.sess_df.merge(self.prod_df).merge(self.usr_df)
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

    def delete_attributes_except_amount(self, frame):
        frame = frame.drop(
            columns=['is_male', 'viewed_num', 'bought/sum', 'frequency', 'recency', 'bought_num'], axis=1)
        return frame

    def delete_customers_with_below_10_items_bought(self, frame):
        return frame[frame["bought_num"] >= 10]

    def generate_processed_files_minimal(self, out1_path):
        path = os.path.join(out1_path, "processed.csv")
        merged = self.merge_dataframes_and_add_attributes()
        final_frame = self.delete_customers_with_below_10_items_bought(merged)

        minimalistic_frame = self.delete_attributes_except_amount(final_frame)
        minimalistic_frame.to_csv(path, index=False)

    def generate_processed_files_maximal(self, out_path):
        path = os.path.join(out_path, "processed.csv")
        merged = self.merge_dataframes_and_add_attributes()
        final_frame = self.delete_customers_with_below_10_items_bought(merged)
        final_frame.to_csv(path, index=False)
