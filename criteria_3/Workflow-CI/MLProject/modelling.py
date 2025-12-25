import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import os

# Dynamically resolve the absolute path to the mlruns directorys
mlruns_path = os.path.abspath("criteria_3/Workflow-CI/MLProject/mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Set the experiment to an existing one (e.g., "1")
mlflow.set_experiment("1")

data_path = "criteria_3/Workflow-CI/MLProject/insurance_preprocessing/insurance_clean.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # Split
    X = df.drop(columns=['charges'])
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Basic Training ---
    # Use Autolog for convenience
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Basic_RandomForest") as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Explicitly log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model logged under run ID: {run.info.run_id}")
        print("Basic Model Training Completed.")
else:
    print("Dataset file not found. Ensure it has been copied.")