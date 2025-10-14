import numpy as np
import pandas as pd


def calculate_distance_in_km(latitude_1, longitude_1, latitude_2, longitude_2):
    earth_radius = 6371.0088  # earth's mean radius in km
    latitude_1 = np.radians(latitude_1)
    longitude_1 = np.radians(longitude_1)
    latitude_2 = np.radians(latitude_2)
    longitude_2 = np.radians(longitude_2)
    distance_lat = latitude_2 - latitude_1
    distance_lon = longitude_2 - longitude_1
    haversine_formula = np.sin(distance_lat / 2) ** 2 + np.cos(latitude_1) * np.cos(latitude_2) * np.sin(distance_lon / 2) ** 2
    distance_haversine = (2 * earth_radius * np.arcsin(np.sqrt(haversine_formula))).astype(np.float32)
    return distance_haversine

def fix_time_features(df, column_name):
    df_time = pd.to_datetime(df[column_name], errors="coerce")
    df = df.loc[df_time.notna()].copy() # find rows with a valid timestamp
    df_time = df_time.loc[df_time.notna()] # drop all invalid timestamp rows
    df["hour_of_day"] = df_time.dt.hour.astype(np.int16) # hours in 0-23
    df["day_of_week"] = df_time.dt.dayofweek.astype(np.int8) # 0 is Monday, 1 is Tuesday and so on
    df["calendar_month"] = df_time.dt.month.astype(np.int8) # values in 1-12
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_rush_hour"] = ((df["hour_of_day"].between(7, 10)) | (df["hour_of_day"].between(16, 19))).astype(np.int8)
    return df

def fix_coordinates(df, pick_up_lat, pick_up_lon, drop_off_lat, drop_off_lon):
    # between 40.4 and 41.1 is approximately the NYC area in latitude
    # between -74.3 and -73.5 is approximately the NYC area in longitude
    lat_okay = df[pick_up_lat].between(40.4, 41.1) & df[drop_off_lat].between(40.4, 41.1) # check if latitude is okay
    lon_okay = df[pick_up_lon].between(-74.3, -73.5) & df[drop_off_lon].between(-74.3, -73.5) # check if longitude is okay
    return df[lat_okay & lon_okay].copy()


def load_NYTaxi_data(path, n_rows=10000):
    df = pd.read_csv(path, nrows=n_rows)
    # get target column, aka trip_duration
    y = df["trip_duration"].astype(np.float32)

    # we delete extreme outliers
    y = y.clip(lower=60, upper=2 * 60 * 60)
    y = np.log1p(y).astype(np.float32)

    # Building the new X dataset by:
    # keeping only columns that we consider are useful, such as vendor_id and pick_up and drop_off attributes
    needed_features = [ "vendor_id", "pickup_datetime", "dropoff_datetime",
        "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude"]
    X = df[[feature for feature in needed_features if feature in df.columns]].copy()

    # fix "time" features from pickup attribute
    if "pickup_datetime" in X.columns:
        X = fix_time_features(X, "pickup_datetime")

    # drop outliers and None values from pick_up and drop_off attribute values of coordinates
    X = fix_coordinates(X, pick_up_lat="pickup_latitude", pick_up_lon="pickup_longitude",
        drop_off_lat="dropoff_latitude", drop_off_lon="dropoff_longitude")

    # compute well-known (haversine, manhattan) distance metrics and drop raw timestamps
    X["haversine_km"] = calculate_distance_in_km(X["pickup_latitude"].values, X["pickup_longitude"].values,
        X["dropoff_latitude"].values, X["dropoff_longitude"].values)
    X["manhattan_km"] = ((X["pickup_latitude"] - X["dropoff_latitude"]).abs() + (X["pickup_longitude"] - X["dropoff_longitude"]).abs()) * 111.0
    X["direction_in_deg"] = np.degrees(np.arctan2(
        np.sin(np.radians(X["dropoff_longitude"] - X["pickup_longitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])),
        np.cos(np.radians(X["pickup_latitude"])) *
        np.sin(np.radians(X["dropoff_latitude"])) -
        np.sin(np.radians(X["pickup_latitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])) *
        np.cos(np.radians(X["dropoff_longitude"] - X["pickup_longitude"]))
    )).astype(np.float32)
    X["direction_in_deg"] = ((X["direction_in_deg"] + 360) % 360).astype(np.float32) # compute direction of the trip in degrees 0-360 -- acts like a compass

    # clean the rest of the features
    if "passenger_count" in X.columns:
        X["passenger_count"] = pd.to_numeric(X["passenger_count"], errors="coerce").fillna(1).clip(1, 6).astype(np.int8)
    if "vendor_id" in X.columns:
        X["vendor_id"] = pd.to_numeric(X["vendor_id"], errors="coerce").fillna(1).astype(np.int8)
    if "store_and_fwd_flag" in X.columns:
        X["store_and_fwd_flag"] = X["store_and_fwd_flag"].astype(str).str.upper().map({"Y": 1, "N": 0}).fillna(
            0).astype(np.int8)

    # drop raw timestamps columns
    for column in ["pickup_datetime", "dropoff_datetime"]:
        if column in X.columns: X.drop(columns=column, inplace=True)

    # drop raw pick_up and drop_off features
    X.drop(columns=["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], inplace=True)

    # cast values of attributes to numeric ones
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for column in X.columns:
        if X[column].dtype == "float64": X[column] = X[column].astype(np.float32)

    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y