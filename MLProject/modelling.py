import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import pathlib

# Dynamically resolve the absolute path to the mlruns directory
mlruns_path = pathlib.Path("mlruns").absolute().as_uri()
mlflow.set_tracking_uri(mlruns_path)

# Automatically create or use the experiment
experiment_name = "1"
mlflow.set_experiment(experiment_name)

data_path = "./insurance_preprocessing/insurance_clean.csv"

# Check if the dataset exists
if not os.path.exists(data_path):
    # Create a dummy dataset if it doesn't exist
    print("Dataset file not found. Creating a dummy dataset...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    dummy_data = {
        "age": [25, 30, 35, 40],
        "sex": ["male", "female", "male", "female"],
        "bmi": [28.5, 24.3, 30.1, 22.7],
        "children": [0, 1, 2, 3],
        "smoker": ["no", "yes", "no", "yes"],
        "charges": [16884.92, 1725.55, 4449.46, 21984.47],
        "loc_northwest": [0, 1, 0, 1],
        "loc_southeast": [1, 0, 1, 0],
        "loc_southwest": [0, 0, 0, 0],
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv(data_path, index=False)
    print(f"Dummy dataset created at: {data_path}")
else:
    df = pd.read_csv(data_path)

# Encode categorical columns
categorical_columns = ["sex", "smoker"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the dataset
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Basic Training ---
with mlflow.start_run(run_name="Basic_RandomForest") as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Explicitly log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model logged under run ID: {run.info.run_id}")
    print("Basic Model Training Completed.")