import pandas as pd
import numpy as np


def load_EPowerConsumption(path, n_rows):
    path = "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/household_power_consumption.txt"

    cols = ["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pd.read_csv(path, sep=";", nrows=n_rows, usecols=cols,
                     na_values="?", low_memory=False)

    # parse datetime (day-first in this dataset)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["dow"] = df["datetime"].dt.dayofweek.astype("int8")
    # df["epoch_s"] = df["datetime"].astype("int64") / 1e9  # alternative single numeric time feature

    y = pd.to_numeric(df["Global_active_power"], errors="coerce")

    X = df[[
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour", "dow"  # or "epoch_s"
    ]].copy()

    mask = df["datetime"].notna() & y.notna() & (~X.isna().any(axis=1))
    X = X.loc[mask].astype(np.float32).reset_index(drop=True)
    y = y.loc[mask].astype(np.float32).reset_index(drop=True)

    print(X.shape, y.shape)
    return X, y