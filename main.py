import numpy as np
import pandas as pd
from datetime import datetime
COLUMNS_POLICY = {"D": "days", "N": "nights", "P": "price"}


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


def apply_booking_date(df):
    return df.apply(lambda row: days_difference(row['booking_datetime'], row['checkin_date']), axis=1)


def clean_hotel_id(df):
    # df["hotel_id"]
    pass


# book date A
# reservation date B
# cancellation date C
# days for free cancellation D
# checkout date E
# formula = (B - A) / D

if __name__ == "__main__":
    np.random.seed(0)
    df = pd.read_csv("agoda_cancellation_train.csv")
    create_cancellation_colunmn(df)
    print(apply_booking_date(df))
    # X = pd.get_dummies(df, columns=['hotel_id'])


    # print(df)