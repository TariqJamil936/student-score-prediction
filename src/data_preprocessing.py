# Handles loading the dataset, cleaning anomalies, imputing missing values, and partitioning into train/test sets.
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path: str) -> pd.DataFrame:
  """Loads student performance data, cleans boundaries, and imputes nulls."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

  df = pd.read_csv(file_path)
  df = df.drop_duplicates()

  # Impute missing numeric columns using median
  numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
  for col in numeric_cols:
    if df[col].isnull().sum() > 0:
      df[col] = df[col].fillna(df[col].median())

  # Impute categorical columns using mode
  categorical_cols = df.select_dtypes(include=["object"]).columns
  for col in categorical_cols:
    if df[col].isnull().sum() > 0:
      df[col] = df[col].fillna(df[col].mode()[0])

  # Sanity checks: realistic constraints
  df = df[(df["Hours_Studied"] >= 0) & (df["Hours_Studied"] <= 168)]
  df = df[(df["Exam_Score"] >= 0) & (df["Exam_Score"] <= 100)]

  return df


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "Exam_Score",
    test_size: float = 0.20,
    random_state: int = 42,
):
  """Splits features and target into train and test splits."""
  if feature_cols is None:
    feature_cols = ["Hours_Studied"]

  X = df[feature_cols]
  y = df[target_col]

  return train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )