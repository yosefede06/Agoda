import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn import ensemble
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
max_depth = 30
COLUMNS_POLICY = {"D": "days", "N": "nights", "P": "price"}
DROP_COLUMNS = [ "hotel_id", "customer_nationality", 'no_of_adults', "no_of_children", "no_of_room",
               'guest_nationality_country_name', 'language','original_payment_currency',
                'request_nonesmoke','request_latecheckin','request_highfloor','request_largebed',
                'request_twinbeds','request_airport','request_earlycheckin', "hotel_brand_code", "hotel_chain_code",
                 'hotel_area_code','hotel_city_code', "h_customer_id"]
DUMMIES_COLUMNS = ['hotel_country_code', 'accommadation_type_name',
                   'original_payment_method','original_payment_type', 'charge_option']
OUT_CANCELLATION_PREDICTION_FILENAME = "agoda_cancellation_prediction.csv"
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
    create a new column that represents the number of weeks from the booking date to the checkin date
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']), axis=1)

def create_for_free_cancelation(df):
    """
    create a new column that represents the number of weeks from the booking date until the end of the free cancelation period
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df = df.drop(df[df["cancellation_policy_code"] == "UNKNOWN"].index)
    return df.apply(lambda row: max(days_difference(row['booking_datetime'], row['checkin_date']) -
                                round((max(item['days'] for item in decode_policy(row['cancellation_policy_code'])) + 1)/7), 0), axis=1)

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
    create a new column that takes the value True if the guest is in the same country than the hostel, False otherwise.
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    return np.where(df['hotel_country_code'] == df['origin_country_code'], 1, 0)


def add_features(df):
    """
    add 4 new features:
    "weeks_to_checkin": Calculated the number of weeks between the booking date and the check-in date, providing insights into the time gap.
    "duration_trip": Computed the duration of the trip by subtracting the check-in date from the checkout date, capturing the length of the stay.
    "same_country": True if the guest is from the booking country, False otherwise
    "time_for_free_cancellation": Determined the remaining time for free cancellation based on the check-in date and hotel's cancellation policy.
    :param df: Dataframe, Design matrix of regression problem
    :return: df
    """
    df["weeks_to_checkin"] = apply_booking_date(df)
    df["duration_trip"] = create_trip_duration(df)
    df["same_country"] = create_guest_same_country_booking(df)
    df["time_for_free_cancellation"] = create_for_free_cancelation(df)
    return df

def create_predicted_selling_amount_column(df):
    return df.apply(lambda row: compute_policy(row["cancellation_policy_code"], row['checkin_date'], row['checkout_date'],
                                              row["cancellation_datetime"], row["original_selling_amount"]), axis=1)

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



def ouput_csv(col_id, y_pred):
    df = pd.DataFrame({'ID': col_id, 'cancellation': y_pred})
    filepath = Path(OUT_CANCELLATION_PREDICTION_FILENAME)
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
    df = df.drop(columns=["origin_country_code", "booking_datetime", "checkin_date", "checkout_date",
                          "hotel_live_date", "cancellation_policy_code"])
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

def compute_policy(policy, checkin_date, checkout_date, cancellation_date, original_selling_amount):
    checkin_date_format = "%Y-%m-%d %H:%M:%S"
    cancellation_date_format = "%Y-%m-%d"
    date2 = datetime.strptime(checkin_date, checkin_date_format)
    date3 = datetime.strptime(checkout_date, checkin_date_format)
    date1 = datetime.strptime(cancellation_date, cancellation_date_format)
    difference = str(date2 - date1)
    num_days_cancellation_to_checkin = int(difference.split(" ")[0])
    best_policy = {}
    best_diff = np.inf
    for pol in decode_policy(policy):
        if abs(num_days_cancellation_to_checkin-pol["days"]) < best_diff:
            best_diff = abs(num_days_cancellation_to_checkin-pol["days"])
            best_policy = pol
    if best_policy["days"] < num_days_cancellation_to_checkin:
        return 0.0
    elif best_policy["nights"] < 0:
        return (best_policy["price"]/100) * original_selling_amount
    else:
        trip_duration = str(date3 - date2)
        trip_duration = int(trip_duration.split(" ")[0])
        return (best_policy["nights"]/trip_duration) * original_selling_amount



def classifier_fit(X, y):
    """
    Apply random forest to classify
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    :return:
    """
    regr = ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
    regr = ensemble.RandomForestRegressor(n_estimators=100, max_depth = max_depth, random_state=0)
    regr.fit(X, y)
    return regr

def classifier_predict(X, regr):
    return regr.predict(X)

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

def regression_fit(train_X, train_y):
    return LinearRegression().fit(X, y)

def clean_data():
    """
    cleans the data and preprocess it
    :return:
    """
    np.random.seed(0)
    df = pd.read_csv("agoda_cancellation_train.csv")
    df = preprocess_data(df)
    y_pred = create_predicted_selling_amount_column(df)
    y = create_cancellation_colunmn(df)
    X = df.drop(columns=["cancellation_datetime"])
    return X, y

def clean_id(train_X, train_y, train_cross_X, train_cross_y):
    col_id = train_cross_X["h_booking_id"]
    train_X.drop(columns=["h_booking_id"])
    train_y.drop(columns=["h_booking_id"])
    train_cross_X.drop(columns=["h_booking_id"])
    train_cross_y.drop(columns=["h_booking_id"])
    return col_id

def apply_model_classification(X, y):
    train_X, train_y, train_cross_X, train_cross_y = split_train_test(X, y)
    col_id = clean_id(train_X, train_y, train_cross_X, train_cross_y)
    regr = classifier_fit(train_X, train_y)
    y_pred = classifier_predict(train_cross_X, regr)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    ouput_csv(col_id, y_pred)
    print(misclassification_error(train_cross_y, y_pred))
    print(metrics.f1_score(y_true=train_cross_y, y_pred=y_pred))

def apply_model_regression(X, y):
    train_X, train_y, train_cross_X, train_cross_y = split_train_test(X, y)
    col_id = clean_id(train_X, train_y, train_cross_X, train_cross_y)
    regr = regression_fit(train_X, train_y)
    y_pred = regr.predict(train_cross_X)


if __name__ == "__main__":
    X, y = clean_data()
    apply_model_classification(X, y)
    apply_model_regression(X, y)

    # apply_model_regression(X, y)