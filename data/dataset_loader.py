import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UciOccupancyDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            features: list,
            label_col: str,
            sequence_length: int = 60,
            include_capacity: bool = False
    ):
        """
        Dataset for occupancy prediction using UCI dataset.

        :param data: Pandas DataFrame with aligned sensor data.
        :param features: List of feature columns to be used for prediction (CO2 and light).
        :param label_col: Column name for the target variable (occupancy).
        :param sequence_length: Length of the input sequences. (e.g., 60 for 1 hour of data with 1 minute intervals)
        :param include_capacity: Whether to include the room capacity as a static feature
        """

        self.sequence_length = sequence_length
        self.label_col = label_col

        # Include room capacity as a contextual feature
        self.features = features + ['room_capacity'] if include_capacity else features

        # Reset index in case previous operations changed row order
        self.data = data.reset_index(drop=True)

        # Convert to float32 numpy array for efficiency
        self.feature_array = self.data[self.features].to_numpy(dtype=np.float32)
        self.label_array = self.data[self.label_col].to_numpy(dtype=np.float32)

    def __len__(self):
        # Subtract sequence length to avoid index out of range
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Extract a window of features
        x_seq = self.feature_array[idx:idx + self.sequence_length].copy()

        # Get occupancy label for the final timestep
        y = self.label_array[idx + self.sequence_length - 1]

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x_seq) # Shape: (sequence_length, num_features)
        y_tensor = torch.tensor(y, dtype=torch.float32) # Shape: (1,)

        return x_tensor, y_tensor


class Lab42Dataset(Dataset):
    def __init__(
            self,
            data,
            features,
            label_col,
            sequence_length=60,
            include_capacity=True
    ):
        """
        Dataset for occupancy prediction using the Lab42 dataset.

        :param data: Pandas DataFrame with aligned sensor data.
        :param features: Feature columns to be used for prediction (e.g., air quality, light).
        :param label_col: Label column name for the target variable (occupancy).
        :param sequence_length: Sequence length for input data (e.g., 60 for 1 hour of data with 1 minute intervals).
        :param include_capacity: Include room capacity as a static feature in the dataset.
        """

        self.seq_len = sequence_length
        self.label_col = label_col
        self.features = features + (["capacity"] if include_capacity else [])

        self.data = data.reset_index(drop=True)

        # Drop rows with NaNs in features or label
        self.data.dropna(subset=self.features + [label_col], inplace=True)

        # Normalize the feature
        if "capacity" in self.data.columns:
            self.data["capacity"] = (self.data["capacity"] - self.data["capacity"].min()) / (
                    self.data["capacity"].max() - self.data["capacity"].min()
            )

        self.feature_array = self.data[self.features].values.astype('float32')
        self.label_array = self.data[self.label_col].values.astype('float32')

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.feature_array[idx:idx+self.seq_len]
        y = self.label_array[idx+self.seq_len-1]

        y = float(y) # Convert to float from Boolean for PyTorch

        x_tensor = torch.tensor(x_seq)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if torch.isnan(x_tensor).any() or torch.isinf(x_tensor).any():
            print("Invalid input found at index", idx)
            print("x_tensor:", x_tensor)
            print("y_tensor:", y_tensor)

        return x_tensor, y_tensor

def load_lab42_from_influxdb(url, token, org, bucket, start, stop, file_path="lab42_sensor_data.csv"):
    """
        Function was previously used to load the Lab42 dataset from InfluxDB using the InfluxDBClient. Data is now loaded from a CSV file.

    :param file_path: File path to the CSV file for Snellius
    :param url: InfluxDB URL
    :param token: InfluxDB token
    :param org: InfluxDB organization
    :param bucket: InfluxDB bucket
    :param start: InfluxDB start time
    :param stop: InfluxDB stop time
    :return: DataFrame with the Lab42 dataset
    """
    #
    # client = InfluxDBClient(url=url, token=token, org=org, timeout=3_600_000)
    # query_api = client.query_api()
    #
    # # Query to get the data from InfluxDB
    # query = f'''
    # from(bucket: "{bucket}")
    # |> range(start: {start}, stop: {stop})
    # |> filter(fn: (r) => r["_measurement"] == "room_data")
    # |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    # |> keep(columns: ["_time", "room_number", "temperature", "airquality", "light", "daylight", "lecture_scheduled", "capacity"])
    # '''
    #
    # # Execute the query and convert to DataFrame
    # results = query_api.query_data_frame(query)
    # results["_time"] = pd.to_datetime(results["_time"])
    #
    # # Rename label to "Occupancy"
    # results.rename(columns={"lecture_scheduled": "Occupancy"}, inplace=True)
    # results["Occupancy"] = results["Occupancy"].astype(int)
    # results["Occupancy"] = results["Occupancy"].astype(np.float32)
    #
    # return results

    df = pd.read_csv(file_path, parse_dates=["_time"])
    df = df[(df["_time"] >= pd.to_datetime(start)) & (df["_time"] <= pd.to_datetime(stop))]
    return df



def add_contextual_features(dataframe, normalize=False):
    """
    Add contextual features to the data.

    :param normalize: Flag to normalize features
    :param dataframe: DataFrame to add features to
    :return: DataFrame with added features
    """

    original_len = len(dataframe)
    dataframe = dataframe.dropna(how="any").copy()  # Drop rows with any missing values
    print(f"Dropped {original_len - len(dataframe)} rows due to NaNs.")

    # Time based features
    dataframe.loc[:, "_time"] = pd.to_datetime(dataframe["_time"])
    dataframe.loc[:, "hour_of_day"] = dataframe["_time"].dt.hour / 23.0 # Normalize to [0, 1]
    dataframe.loc[:, "is_weekend"] = (dataframe["_time"].dt.dayofweek >= 5).astype(int) # 0 for weekday, 1 for weekend

    # Delta features
    if "airquality" in dataframe.columns:
        dataframe.loc[:, "airquality_delta"] = dataframe["airquality"].diff().fillna(0) # Fill NaN with 0
        dataframe.loc[:, "airquality_trend"] = dataframe["airquality_delta"].rolling(window=5, min_periods=1).mean() # Rolling mean for trend

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        dataframe[["airquality", "airquality_delta", "airquality_trend", "light"]] = scaler.fit_transform(
            dataframe[["airquality", "airquality_delta", "airquality_trend", "light"]]
        )

    return dataframe

