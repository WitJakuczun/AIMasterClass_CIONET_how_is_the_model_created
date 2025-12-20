import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np


class DataSplitter:
    """
    A class to handle various data splitting tasks for machine learning experiments.
    """

    def split_train_test(self, df: pl.DataFrame, test_size: float, stratify_on: str):
        """
        Splits a dataframe into a training set and a test set.

        Args:
            df (pl.DataFrame): The input dataframe.
            test_size (float): The proportion of the dataset to allocate to the test split.
            stratify_on (str): The column to use for stratified splitting.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame | None]: A tuple containing the training dataframe
                                                     and the test dataframe. If test_size is 0,
                                                     the test dataframe will be None.
        """
        if test_size > 0:
            # We convert to pandas temporarily because sklearn's train_test_split
            # handles stratification and indexing more naturally with it.
            # Alternatively we could use indices.
            indices = np.arange(df.height)
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=42,
                stratify=df[stratify_on].to_numpy(),
            )
            return df[train_idx], df[test_idx]
        else:
            return df, None

    def split_cv(self, df: pl.DataFrame, n_splits: int, stratify_on: str):
        """
        Generates cross-validation folds from a dataframe, splitting each fold into train, validation, and test sets.

        Args:
            df (pl.DataFrame): The input dataframe.
            n_splits (int): The number of folds.
            stratify_on (str): The column to use for stratified splitting.

        Yields:
            tuple[int, pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing the fold number,
                                                                   the training dataframe,
                                                                   the validation dataframe,
                                                                   and the test dataframe for the fold.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        indices = np.arange(df.height)
        for fold, (train_index, val_test_index) in enumerate(
            skf.split(indices, df[stratify_on].to_numpy())
        ):
            train_df = df[train_index]
            val_test_df = df[val_test_index]

            # Split the validation/test set into validation and test sets
            val_test_indices = np.arange(val_test_df.height)
            val_idx, test_idx = train_test_split(
                val_test_indices,
                test_size=0.5,
                random_state=42,
                stratify=val_test_df[stratify_on].to_numpy(),
            )

            yield fold, train_df, val_test_df[val_idx], val_test_df[test_idx]

    def split_train_val(
        self, df: pl.DataFrame, val_size: float, test_size: float, stratify_on: str
    ):
        """
        Splits a dataframe into a training set, a validation set, and a test set.

        Args:
            df (pl.DataFrame): The input dataframe.
            val_size (float): The proportion of the dataset to allocate to the validation split.
            test_size (float): The proportion of the dataset to allocate to the test split.
            stratify_on (str): The column to use for stratified splitting.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing the training, validation,
                                                              and test dataframes.
        """
        indices = np.arange(df.height)
        train_idx, val_test_idx = train_test_split(
            indices,
            test_size=(val_size + test_size),
            random_state=42,
            stratify=df[stratify_on].to_numpy(),
        )

        train_df = df[train_idx]
        val_test_df = df[val_test_idx]

        # Calculate the proportion of the remaining data that should be the test set
        relative_test_size = test_size / (val_size + test_size)

        val_test_indices = np.arange(val_test_df.height)
        val_idx, test_idx = train_test_split(
            val_test_indices,
            test_size=relative_test_size,
            random_state=42,
            stratify=val_test_df[stratify_on].to_numpy(),
        )

        return train_df, val_test_df[val_idx], val_test_df[test_idx]
