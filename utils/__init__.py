import pandas as pd
from rectools import Columns


def read_split_rating_dataset(path, split_dt="1998-03-01"):
    """
    Read and split rating dataset into train and test set

    :param path: path to rating dataset
    :param split_dt: date, after which all ratings are in test set
    :return: train and test set
    """
    df = pd.read_csv(path)
    df.datetime = pd.to_datetime(df.datetime)

    split_dt = pd.Timestamp(split_dt)
    df_train = df.loc[df["datetime"] < split_dt]
    df_test = df.loc[df["datetime"] >= split_dt]

    # Remove non-intersected users and items in test set
    df_test = df_test.loc[df_test[Columns.User].isin(df_train[Columns.User])]
    df_test = df_test.loc[df_test[Columns.Item].isin(df_train[Columns.Item])]

    return df_train, df_test


def read_split_rating_dataset_with_features(rating_path, users_path, movies_path, split_dt="1998-03-01"):
    """
    Read and split rating dataset into train and test set and read user and item features

    :param rating_path: path to rating dataset
    :param users_path: path to users dataset
    :param movies_path: path to movies dataset
    :param split_dt: date, after which all ratings are in test set
    :return: train rating, user and item features, test rating set
    """
    df_train, df_test = read_split_rating_dataset(rating_path, split_dt)

    users = pd.read_csv(users_path)
    user_features_frames = []
    for feature in ["age", "gender", "occupation"]:
        feature_frame = users.reindex(columns=["user_id", feature])
        feature_frame.columns = ["id", "value"]
        feature_frame["feature"] = feature
        user_features_frames.append(feature_frame)
    user_features = pd.concat(user_features_frames)
    user_features_train = user_features.loc[user_features["id"].isin(df_train[Columns.User])]

    items = pd.read_csv(movies_path)
    item_features_frames = []
    for feature in items.drop(columns=['movie_id', 'movie_title', 'release_date']):
        feature_frame = items.reindex(columns=["movie_id", feature])
        feature_frame.columns = ["id", "value"]
        feature_frame["feature"] = feature
        item_features_frames.append(feature_frame)
    item_features = pd.concat(item_features_frames)
    item_features_train = item_features.loc[item_features["id"].isin(df_train[Columns.Item])]

    return df_train, user_features_train, item_features_train, df_test
