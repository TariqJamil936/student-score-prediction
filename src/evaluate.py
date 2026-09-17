# Loads the saved models, computes validation metrics on test data ($R^2$, MAE, RMSE), prints a summary table, and plots the regression fit.
import os
import pickle
from data_preprocessing import load_and_clean_data, prepare_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models():
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  # Resolve data path
  data_path = os.path.join(
      base_dir, "data", "cleaned_StudentPerformanceFactors.csv"
  )
  if not os.path.exists(data_path):
    data_path = os.path.join(
        base_dir, "data", "StudentPerformanceFactors.csv"
    )

  df = load_and_clean_data(data_path)
  _, X_test, _, y_test = prepare_data(df, feature_cols=["Hours_Studied"])

  # Load trained models
  models_dir = os.path.join(base_dir, "models")
  linear_path = os.path.join(models_dir, "linear_model.pkl")
  poly_path = os.path.join(models_dir, "poly_model.pkl")

  if not os.path.exists(linear_path) or not os.path.exists(poly_path):
    raise FileNotFoundError(
        "Trained models not found. Run `python src/train.py` first."
    )

  with open(linear_path, "rb") as f:
    linear_model = pickle.load(f)

  with open(poly_path, "rb") as f:
    poly_model = pickle.load(f)

  # Predictions
  y_pred_linear = linear_model.predict(X_test)
  y_pred_poly = poly_model.predict(X_test)

  # Metric evaluation
  def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return round(mae, 2), round(rmse, 2), round(r2, 4)

  mae_lin, rmse_lin, r2_lin = calc_metrics(y_test, y_pred_linear)
  mae_poly, rmse_poly, r2_poly = calc_metrics(y_test, y_pred_poly)

  print("\n" + "=" * 45)
  print("         MODEL PERFORMANCE COMPARISON")
  print("=" * 45)
  results_df = pd.DataFrame({
      "Model": ["Linear Regression (Deg 1)", "Polynomial Regression (Deg 2)"],
      "MAE": [mae_lin, mae_poly],
      "RMSE": [rmse_lin, rmse_poly],
      "R² Score": [r2_lin, r2_poly],
  })
  print(results_df.to_string(index=False))
  print("=" * 45 + "\n")

  # Visualization
  plt.figure(figsize=(9, 5))
  plt.scatter(
      X_test,
      y_test,
      color="#1f77b4",
      alpha=0.4,
      label="Actual Scores (Test Set)",
  )

  # Sort points for plotting smooth curves
  sort_idx = np.argsort(X_test["Hours_Studied"].values)
  sorted_hours = X_test["Hours_Studied"].values[sort_idx]

  plt.plot(
      sorted_hours,
      y_pred_linear[sort_idx],
      color="#d62728",
      linewidth=2.5,
      label=f"Linear Fit (R² = {r2_lin:.3f})",
  )
  plt.plot(
      sorted_hours,
      y_pred_poly[sort_idx],
      color="#2ca02c",
      linewidth=2.5,
      linestyle="--",
      label=f"Polynomial Fit (R² = {r2_poly:.3f})",
  )

  plt.title(
      "Test Set Evaluation: Study Hours vs. Exam Score",
      fontsize=13,
      weight="bold",
  )
  plt.xlabel("Hours Studied", fontsize=11)
  plt.ylabel("Exam Score", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  evaluate_models()