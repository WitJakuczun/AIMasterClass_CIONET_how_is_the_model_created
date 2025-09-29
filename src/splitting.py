import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

class DataSplitter:
    """
    A class to handle various data splitting tasks for machine learning experiments.
    """

    def split_train_test(self, df: pd.DataFrame, test_size: float, stratify_on: str):
        """
        Splits a dataframe into a training set and a test set.

        Args:
            df (pd.DataFrame): The input dataframe.
            test_size (float): The proportion of the dataset to allocate to the test split.
            stratify_on (str): The column to use for stratified splitting.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: A tuple containing the training dataframe
                                                     and the test dataframe. If test_size is 0,
                                                     the test dataframe will be None.
        """
        if test_size > 0:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[stratify_on])
            return train_df, test_df
        else:
            return df, None

    def split_cv(self, df: pd.DataFrame, n_splits: int, stratify_on: str):
        """
        Generates cross-validation folds from a dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.
            n_splits (int): The number of folds.
            stratify_on (str): The column to use for stratified splitting.

        Yields:
            tuple[int, pd.DataFrame, pd.DataFrame]: A tuple containing the fold number,
                                                   the training dataframe for the fold,
                                                   and the validation dataframe for the fold.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(skf.split(df, df[stratify_on])):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]
            yield fold, train_df, val_df

    def split_train_val(self, df: pd.DataFrame, val_size: float, stratify_on: str):
        """
        Splits a dataframe into a training set and a validation set.

        Args:
            df (pd.DataFrame): The input dataframe.
            val_size (float): The proportion of the dataset to allocate to the validation split.
            stratify_on (str): The column to use for stratified splitting.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training dataframe
                                              and the validation dataframe.
        """
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=42, stratify=df[stratify_on])
        return train_df, val_df