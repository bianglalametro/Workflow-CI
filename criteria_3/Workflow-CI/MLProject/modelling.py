import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import pathlib
import os

# --- MLflow tracking setup ---
mlruns_path = pathlib.Path("mlruns").absolute()
mlflow.set_tracking_uri(f"file://{mlruns_path}")  # Use file:// format
mlflow.set_experiment("Insurance Cost Prediction Experiment")

# --- Load or create dataset ---
base_dir = pathlib.Path(__file__).parent.absolute()
data_path = base_dir / "insurance_preprocessing" / "insurance_clean.csv"

if not data_path.exists():
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "sex": ["male", "female", "male", "female"],
        "bmi": [28.5, 24.3, 30.1, 22.7],
        "children": [0, 1, 2, 3],
        "smoker": ["no", "yes", "no", "yes"],
        "charges": [16884.92, 1725.55, 4449.46, 21984.47],
        "loc_northwest": [0, 1, 0, 1],
        "loc_southeast": [1, 0, 1, 0],
        "loc_southwest": [0, 0, 0, 0],
    })
    df.to_csv(data_path, index=False)
else:
    df = pd.read_csv(data_path)

# Encode categorical columns
for col in ["sex", "smoker"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split dataset
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train and log model as artifact ---
with mlflow.start_run(run_name="RandomForest_Insurance") as run:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log to run artifacts folder
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # logs under artifacts/model
        registered_model_name=None
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model logged at: {run.info.artifact_uri}/model")

# Debug
print(f"MLruns path: {mlruns_path}")
print(f"Current working dir: {os.getcwd()}")
