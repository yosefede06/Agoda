import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn import ensemble
import matplotlib.pyplot as plt

COLUMNS_POLICY = {"D": "days", "N": "nights", "P": "price"}
DROP_COLUMNS = [ "hotel_id", "customer_nationality", 'no_of_adults', "no_of_children", "no_of_room",
               'guest_nationality_country_name', 'language','original_payment_currency',
                'request_nonesmoke','request_latecheckin','request_highfloor','request_largebed',
                'request_twinbeds','request_airport','request_earlycheckin', "hotel_brand_code", "hotel_chain_code",
                 'hotel_area_code','hotel_city_code', "h_customer_id"]
DUMMIES_COLUMNS = ['hotel_country_code', 'accommadation_type_name',
                   'original_payment_method','original_payment_type', 'charge_option']

def drop_useless_columns(df):
    df = df.drop(columns=DROP_COLUMNS)
    return df


def apply_booking_date(df):
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']), axis=1)

def create_for_free_cancelation(df):
    df = df.drop(df[df["cancellation_policy_code"] == "UNKNOWN"].index)
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']) -
                                round((max(item['days'] for item in decode_policy(row['cancellation_policy_code'])) + 1)/7), axis=1)


def create_trip_duration(df):
    return df.apply(lambda row: days_difference(row['checkin_date'], row['checkout_date']), axis=1)


def create_guest_same_country_booking(df):
    return np.where(df['hotel_country_code'] == df['origin_country_code'], 1, 0)


def add_features(df):
    df["weeks_to_checkin"] = apply_booking_date(df)
    df["duration_trip"] = create_trip_duration(df)
    df["same_country"] = create_guest_same_country_booking(df)
    df["time_for_free_cancellation"] = create_for_free_cancelation(df)
    return df


def create_cancellation_colunmn(df):
    date_cancellation = df["cancellation_datetime"]
    return date_cancellation.isnull().astype(int)


def decode_policy(code):
    data = []
    policies = code.split("_")
    for pol in policies:
        new_data = {}
        for col in COLUMNS_POLICY:
            if col in pol:
                temp = pol.split(col)
                new_data[COLUMNS_POLICY[col]] = int(temp[0])
                pol = "".join(temp[1:])
            else:
                new_data[COLUMNS_POLICY[col]] = -1
        data.append(new_data)
    return data


def days_difference(booking_date_str, check_in_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"

    booking_date = datetime.strptime(booking_date_str, date_format)
    check_in_date = datetime.strptime(check_in_date_str, date_format)

    difference = check_in_date - booking_date
    return round((difference.days + 1) / 7)


def transform_to_binary(df):
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    df["is_user_logged_in"] = df["is_user_logged_in"].astype(int)
    return df


def preprocess_data(df):
    df = drop_useless_columns(df)
    df = add_features(df)
    df = df.drop(columns=["origin_country_code", "booking_datetime", "checkin_date", "checkout_date", "hotel_live_date", "h_booking_id"])
    df = classify_columns(df)
    df = transform_to_binary(df)
    return df


def classify_columns(df):
    return pd.get_dummies(df, columns=DUMMIES_COLUMNS, dtype=float)


def classifier_fit(X, y):
    regr = ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
    regr.fit(X, y)
    return regr

def classifier_predict(X, regr):
    return regr.predict(X)

def classifier_loss():
    pass

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = 0.8):
    train_X = X.sample(frac=train_proportion)
    test_X = X.loc[X.index.difference(train_X.index)]
    return train_X, y.loc[train_X.index], test_X, y.loc[test_X.index]


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    y_res = y_true - y_pred
    false_response = np.count_nonzero(y_res)
    if normalize:
        return false_response / np.size(y_res)
    return false_response

def clean_data():
    np.random.seed(0)
    df = pd.read_csv("agoda_cancellation_train.csv")
    df = preprocess_data(df)
    y = create_cancellation_colunmn(df)
    X = df.drop(columns=["cancellation_datetime"])
    return X, y

def apply_model(X, y):
    train_X, train_y, train_cross_X, max_depth = max_depth, train_cross_y = split_train_test(X, y)
    create_cancellation_colunmn(df)
    regr = classifier_fit(train_X, train_y)
    y_pred = classifier_predict(train_cross_X, regr)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    print(misclassification_error(train_cross_y, y_pred))

max_depth = 30
if __name__ == "__main__":
    X, y = clean_data()
    a = 2
    # apply_model(X, y)





    # print(apply_booking_date(df))


    # print(df)