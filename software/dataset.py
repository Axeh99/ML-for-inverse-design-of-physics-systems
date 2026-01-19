import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataSet:
    """
    Class for loading and preprocessing tabular data from an .xlsx file.

    Parameters
    ----------
    file_path : str
        Path to the .xlsx file.
    target_columns : List[str]
        Names of the columns that will be used as targets (y).
        Can contain one or multiple target variables.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
    """

    file_path: str
    seed: int
    feature_columns: List[str]
    target_columns: List[str]             

    test_size: float = 0.2
    drop_columns: List[str] = field(default_factory=list)

    raw_df: Optional[pd.DataFrame] = field(default=None, init=False)
    df: Optional[pd.DataFrame] = field(default=None, init=False)

    X_full: Optional[pd.DataFrame] = field(default=None, init=False)
    y_full: Optional[pd.DataFrame] = field(default=None, init=False)

    X_train: Optional[pd.DataFrame] = field(default=None, init=False)
    X_test: Optional[pd.DataFrame] = field(default=None, init=False)
    y_train: Optional[pd.DataFrame] = field(default=None, init=False)
    y_test: Optional[pd.DataFrame] = field(default=None, init=False)



    _categorical_cols: Optional[List[str]] = field(default=None, init=False)
    _numerical_cols: Optional[List[str]] = field(default=None, init=False)


    def load(self) -> None:
        raw = pd.read_excel(self.file_path).replace("−", "-", regex=True)

        required = self.feature_columns + self.target_columns
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = raw[required].copy()

        for c in self.target_columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        self.raw_df = raw
        self.df = df

    def check_integrity(self) -> dict:
        """
        Perform basic integrity checks on the current dataframe.

        Returns
        -------
        report : dict
            Dictionary with basic integrity information:
            - shape
            - dtypes
            - missing_values_per_column
        """
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call 'load()' first.")

        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values_per_column": self.df.isna().sum().to_dict(),
        }
    
    def drop_missing(self, axis: int = 0, how: str = "any") -> None:
        """
        Remove rows or columns with missing values.

        Parameters
        ----------
        axis : int, optional
            0 to drop rows, 1 to drop columns.
        how : {'any', 'all'}, optional
            'any' drops if any NaN present, 'all' drops only if all NaN.
        """
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call 'load()' first.")

        required = self.feature_columns + self.target_columns
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = self.df.dropna(subset=required, how=how)
        self.X_full = self.df[self.feature_columns]
        self.y_full = self.df[self.target_columns].to_numpy() if len(self.target_columns) > 1 else self.df[self.target_columns[0]].to_numpy()

    def fill_missing(self, strategy: str = "mean", value: Optional[float] = None) -> None:
        """
        Fill missing values according to a simple strategy.

        Parameters
        ----------
        strategy : {'mean', 'median', 'value'}, optional
            Strategy for numerical columns. For 'value', the 'value' parameter is used.
        value : float, optional
            Value used when strategy == 'value'.
        """
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call 'load()' first.")

        if strategy not in {"mean", "median", "value"}:
            raise ValueError("strategy must be one of {'mean', 'median', 'value'}.")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if strategy == "mean":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == "median":
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:  # 'value'
            if value is None:
                raise ValueError("A 'value' must be provided when strategy == 'value'.")
            self.df[numeric_cols] = self.df[numeric_cols].fillna(value)


    def _split_X_y(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.df is None:
            raise ValueError("DataFrame is not loaded or preprocessed. Call 'load()' first.")

        X = self.df[self.feature_columns].copy()
        y_df = self.df[self.target_columns].copy()

        if len(self.target_columns) == 1:
            y = y_df.iloc[:, 0].to_numpy()
        else:
            y = y_df.to_numpy()

        return X, y
    
    def create_train_test_split(self) -> None:
        """
        Create train/test split and store as numpy arrays.

        Notes
        -----
        - X_train and X_test are stored as 2D numpy arrays (n_samples, n_features).
        - y_train and y_test are stored as 2D numpy arrays even if there is only one target,
          i.e. shape (n_samples, n_targets).
        """
        X_df, y = self._split_X_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_df,
            y,
            test_size=self.test_size,
            random_state=self.seed,
        )

        self._categorical_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        self._numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()

    def build_preprocessor(self) -> ColumnTransformer:
        if self._categorical_cols is None or self._numerical_cols is None:
            raise ValueError("Call create_train_test_split() first.")

        ohe_kwargs = {"handle_unknown": "ignore"}
        try:
            ohe = OneHotEncoder(**ohe_kwargs, sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(**ohe_kwargs, sparse=False)

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True), self._numerical_cols),
                ("cat", ohe, self._categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

    def make_single_row(self, row: Dict[str, Any]) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError("feature_columns not set.")
        out = {c: row.get(c, np.nan) for c in self.feature_columns}
        return pd.DataFrame([out])

    def summary(self) -> None:
        if self.df is None:
            print("Data not loaded.")
            return

        print("Current dataframe shape:", self.df.shape)
        print("Feature columns:", self.feature_columns)
        print("Target columns:", self.target_columns)
        if self.X_train is not None:
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
        else:
            print("Train/test split not yet created.")