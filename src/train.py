# Trains the baseline Linear Regression model, fits the polynomial model, and exports the trained baseline model to disk.
import os
import pickle
from data_preprocessing import load_and_clean_data, prepare_data
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def train_models():
  # Resolve base project directory
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  # Priority: clean dataset, fallback to raw dataset
  data_path = os.path.join(
      base_dir, "data", "cleaned_StudentPerformanceFactors.csv"
  )
  if not os.path.exists(data_path):
    data_path = os.path.join(
        base_dir, "data", "StudentPerformanceFactors.csv"
    )

  print(f"[1/4] Loading and cleaning dataset from: {data_path}")
  df = load_and_clean_data(data_path)

  # Prepare Univariate Split (Hours_Studied -> Exam_Score)
  X_train, X_test, y_train, y_test = prepare_data(
      df, feature_cols=["Hours_Studied"]
  )

  # 1. Baseline Linear Regression
  print("[2/4] Training baseline Linear Regression model...")
  linear_model = LinearRegression()
  linear_model.fit(X_train, y_train)

  print(f"      -> Slope (Weight) : {linear_model.coef_[0]:.4f}")
  print(f"      -> Intercept      : {linear_model.intercept_:.4f}")

  # 2. Polynomial Regression (Degree 2)
  print("[3/4] Training Polynomial Regression (Degree 2)...")
  poly_model = make_pipeline(
      PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
  )
  poly_model.fit(X_train, y_train)

  # 3. Save Model Artifacts
  models_dir = os.path.join(base_dir, "models")
  os.makedirs(models_dir, exist_ok=True)

  linear_model_path = os.path.join(models_dir, "linear_model.pkl")
  with open(linear_model_path, "wb") as f:
    pickle.dump(linear_model, f)

  poly_model_path = os.path.join(models_dir, "poly_model.pkl")
  with open(poly_model_path, "wb") as f:
    pickle.dump(poly_model, f)

  print(f"[4/4] Model artifacts saved to '{models_dir}/'")


if __name__ == "__main__":
  train_models()