import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn import ensemble
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import joblib



max_depth = 30
COLUMNS_POLICY = {"D": "days", "N": "nights", "P": "price"}
DROP_COLUMNS = [ "hotel_id", "customer_nationality", 'no_of_adults', "no_of_children", "no_of_room",
               'guest_nationality_country_name', 'language','original_payment_currency',
                'request_nonesmoke','request_latecheckin',
                 'request_highfloor',
                 'request_largebed',
                'request_twinbeds',
                 'request_airport',
                 'request_earlycheckin',
                 "hotel_brand_code",
                 "hotel_chain_code",
                 'hotel_area_code','hotel_city_code', "h_customer_id"]
DUMMIES_COLUMNS = ['hotel_country_code', 'accommadation_type_name',
                   'original_payment_method','original_payment_type', 'charge_option']

OUT_CANCELLATION_PREDICTION_FILENAME =  "1/predictions/agoda_cost_of_cancellation.csv"


def drop_useless_columns(df):
    """
    Removes irrelevant or non-correlated columns (s.t "hotel_id", "customer_nationality")
    that were not expected to significantly impact cancellation predictions.
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df = df.drop(columns=DROP_COLUMNS)
    return df


def apply_booking_date(df):
    """
    create a new column that represents the number of weeks from
     the booking date to the checkin date
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']), axis=1)


def create_for_free_cancelation(df):
    """
    create a new column that represents the number of weeks from the booking
     date until the end of the free cancelation period
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df["cancellation_policy_code"] = df["cancellation_policy_code"].replace("UNKNOWN", "0")
    # df = df[df["cancellation_policy_code"] != "UNKNOWN"]
    return df.apply(lambda row: max(days_difference(row['booking_datetime'], row['checkin_date']) -
                                round((max(item['days'] for item in decode_policy(row['cancellation_policy_code']))
                                       + 1)/7), 0), axis=1)


def create_trip_duration(df):
    """
    create a new column that represents the trip duration (in weeks - upper rounded)
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return df.apply(lambda row: days_difference(row['checkin_date'], row['checkout_date']), axis=1)


def create_guest_same_country_booking(df):
    """
    create a new column that takes the value True if the guest is
    in the same country than the hostel, False otherwise.
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return np.where(df['hotel_country_code'] == df['origin_country_code'], 1, 0)


def add_features(df):
    """
    add 4 new features:
    "weeks_to_checkin": Calculated the number of weeks between the booking date and the check-in date, providing
     insights into the time gap.
    "duration_trip": Computed the duration of the trip by subtracting the check-in date from the checkout date,
    capturing the length of the stay.
    "same_country": True if the guest is from the booking country, False otherwise
    "time_for_free_cancellation": Determined the remaining time for free cancellation based
    on the check-in date and hotel's cancellation policy.
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df["weeks_to_checkin"] = apply_booking_date(df)
    df["duration_trip"] = create_trip_duration(df)
    df["same_country"] = create_guest_same_country_booking(df)
    df["time_for_free_cancellation"] = create_for_free_cancelation(df)
    return df


def create_cancellation_colunmn(df):
    """
    create the cancellation column
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    date_cancellation = df["cancellation_datetime"]
    return date_cancellation.isnull().astype(int)


def decode_policy(code):
    """
    decode the policy column into list of dictionaries with day, night and price of cancellation
    :param code: the policy string
    :return:
    """
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
    """
    computes the difference between two dates in weeks (upper rounded)
    :param booking_date_str: the first date
    :param check_in_date_str: the second date
    :return: the number of weeks (upper rounded) between the two dates
    """
    date_format = "%Y-%m-%d %H:%M:%S"

    booking_date = datetime.strptime(booking_date_str, date_format)
    check_in_date = datetime.strptime(check_in_date_str, date_format)

    difference = check_in_date - booking_date
    return round((difference.days + 1) / 7)


def transform_to_binary(df):
    """
    transform True/False features to binary values
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    df["is_user_logged_in"] = df["is_user_logged_in"].astype(int)
    return df


def ouput_csv(col_id, y_pred, name_file = OUT_CANCELLATION_PREDICTION_FILENAME, name_column='cancellation'):
    df = pd.DataFrame({'ID': col_id, 'cancellation': y_pred})
    filepath = Path(name_file)
    df.to_csv(filepath, index=False)


def preprocess_data(df):
    """
    preprocess the data:
    - Removes irrelevant or non-correlated columns
    - Introduces new features
    - Converts categorical variables into dummy variables
    - Transforms true/false columns into binary values
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df = drop_useless_columns(df)
    df = add_features(df)
    df = classify_columns(df)
    df = transform_to_binary(df)
    return df


def classify_columns(df):
    """
    Converts categorical variables into dummy variables
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return pd.get_dummies(df, columns=DUMMIES_COLUMNS, dtype=float)


def classifier_fit(X, y):
    """
    Apply random forest to classify
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    :return:
    """
    regr = ensemble.RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=0)
    regr.fit(X, y)
    joblib.dump(regr, 'weights1.txt')


def classifier_predict(X, weights):
    # regr = ensemble.RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=0)
    loaded_model = joblib.load('weights1.txt')
    # regr.coef_ = weights
    return loaded_model.predict(X)


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


def clean_data_classifiers(df, train=True, cols_train=None):
    """
    cleans the data and preprocess it
    :return:
    """
    np.random.seed(0)

    df = preprocess_data(df)
    if not train:
        df = df.reindex(columns=cols_train, fill_value=0)
    if train:
        df = df.drop(columns=["origin_country_code",
                          "booking_datetime",
                          "checkin_date",
                          "checkout_date",
                          "hotel_live_date",
                          "cancellation_policy_code"])
    if train:
        y = create_cancellation_colunmn(df)
        X = df.drop(columns=["cancellation_datetime"])
        return X, y
    return df


def clean_id(train_X, train_y, train_cross_X, train_cross_y):
    col_id = train_cross_X["h_booking_id"]
    train_X = train_X.drop(columns=["h_booking_id"])
    train_y = train_y.drop(columns=["h_booking_id"])
    train_cross_X = train_cross_X.drop(columns=["h_booking_id"])
    train_cross_y = train_cross_y.drop(columns=["h_booking_id"])
    return train_X, train_y, train_cross_X, train_cross_y, col_id


def apply_model_classification(train_X, train_y, train_cross_X, train_cross_y):
    # train_X, train_y, train_cross_X, train_cross_y = split_train_test(X, y)
    train_X, train_y, train_cross_X, train_cross_y, col_id = clean_id(train_X, train_y, train_cross_X, train_cross_y)
    regr = classifier_fit(train_X, train_y)
    y_pred = classifier_predict(train_cross_X, regr)
    ouput_csv(col_id, y_pred)
    print(misclassification_error(train_cross_y, y_pred))
    print(metrics.f1_score(y_true=train_cross_y, y_pred=y_pred))
    return regr



def output_csv_test(train_X, train_y, train_cross_X):
    col_id = train_cross_X["h_booking_id"]
    train_X = train_X.drop(columns=["h_booking_id"])
    train_y = train_y.drop(columns=["h_booking_id"])
    train_cross_X = train_cross_X.drop(columns=["h_booking_id"])
    classifier_fit(train_X, train_y)
    y_pred = classifier_predict(train_cross_X, None)
    ouput_csv(col_id, y_pred)

def task_1(test_data, train_data_pd):
    X_cancel_train, y_cancel_train = clean_data_classifiers(df=train_data_pd, train=True)
    X_cancel_test = clean_data_classifiers(df=test_data, train=False, cols_train=X_cancel_train.columns)
    output_csv_test(X_cancel_train, y_cancel_train, X_cancel_test)
