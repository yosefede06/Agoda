import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
COLUMNS_POLICY = {"D": "days", "N": "nights", "P": "price"}
DROP_COLUMNS = [ "hotel_id", "customer_nationality", 'no_of_adults', "no_of_children", "no_of_room",
               'guest_nationality_country_name', 'language','original_payment_currency',
                'request_nonesmoke','request_latecheckin','request_highfloor','request_largebed',
                'request_twinbeds','request_airport','request_earlycheckin', "hotel_brand_code", "hotel_chain_code", 'hotel_area_code','hotel_city_code']
DUMMIES_COLUMNS = ['hotel_country_code', 'accommadation_type_name',
                   'original_payment_method','original_payment_type']

def drop_useless_columns(df):
    df = df.drop(columns=DROP_COLUMNS)
    return df


def drop_null_columns(df, threshold=0.5):
    null_counts = {}
    not_null_counts = {}
    for column in df.columns:
        null_counts[column] = df[column].isnull().sum()
        not_null_counts[column] = df[column].notnull().sum()


    columns_to_drop = []
    for column in df.columns:
        print(column)
        print(float(null_counts[column] / (null_counts[column] + not_null_counts[column])) * 100)
        if float(null_counts[column] / (null_counts[column] + not_null_counts[column])) > threshold:
            columns_to_drop.append(column)
    df = df.drop(columns=columns_to_drop)
    return df


def apply_booking_date(df):
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']), axis=1)


def create_trip_duration(df):
    return df.apply(lambda row: days_difference(row['checkin_date'], row['checkout_date']), axis=1)


def create_guest_same_country_booking(df):
    return np.where(df['hotel_country_code'] == df['origin_country_code'], 1, 0)


def add_features(df):
    df["date_to_checkin"] = apply_booking_date(df)
    df["duration_trip"] = create_trip_duration(df)
    df["same_country"] = create_guest_same_country_booking(df)
    return df


def create_cancellation_colunmn(df):
    date_cancellation = df["cancellation_datetime"]
    df["cancellation"] = date_cancellation.isnull().astype(int)


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
    return difference.days


def change_charge_option(df):
    mapping = {"Pay Now": 0, "Pay Later": 1}


    df["charge_option"] = np.vectorize(mapping.get)(df["charge_option"])



# book date A
# reservation date B
# cancellation date C
# days for free cancellation D
# checkout date E
# formula = (B - A) / D


def preprocess_data(df):
    df = drop_useless_columns(df)
    df = add_features(df)
    return df

def classify_columns(df):
    return pd.get_dummies(df, columns=DUMMIES_COLUMNS, dtype=float)


if __name__ == "__main__":
    np.random.seed(0)
    df = pd.read_csv("agoda_cancellation_train.csv")
    df = preprocess_data(df)
    # create_cancellation_colunmn(df)
    # print(apply_booking_date(df))
    # X = pd.get_dummies(df, columns=['hotel_id'])
    df = classify_columns(df)
    print(drop_null_columns(df))
    print(df)
    print(df.shape)


    # print(df)