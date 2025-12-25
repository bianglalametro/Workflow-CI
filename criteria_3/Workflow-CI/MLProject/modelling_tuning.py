import matplotlib
matplotlib.use('Agg') 

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib  # For saving the model

dagshub.init(repo_owner='bianglalametro', repo_name='Eksperimen_SML_AlthafRafianto', mlflow=True)

mlflow.set_experiment("Insurance_Tuning_Model")

data_path = "insurance_preprocessing/insurance_clean.csv"


print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

target_col = 'charges'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators_range = np.linspace(50, 500, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_score = -np.inf
best_run_id = None

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        run_name = f"RF_est{n_estimators}_depth{max_depth}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Training: {run_name}...")
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse) 
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MAPE", mape)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.tight_layout()
            plt.savefig("actual_vs_predicted.png")
            mlflow.log_artifact("actual_vs_predicted.png") # Upload ke DagsHub
            plt.close()

            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            feat_importances.nlargest(10).plot(kind='barh', color='teal')
            plt.title("Top 10 Feature Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png") # Upload ke DagsHub
            plt.close()

            mlflow.sklearn.log_model(
                model, 
                "model", 
                pip_requirements=["scikit-learn", "pandas", "numpy"]
            )

            if r2 > best_score:
                best_score = r2
                best_run_id = run.info.run_id
                print(f"  --> New Best Model Found! R2: {r2:.4f}")

# --- Hyperparameter Tuning ---
mlflow.set_experiment("Insurance_Tuning_Model")

with mlflow.start_run(run_name="Tuning_RandomForest"):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mse", mse)

    # Save the best model as a .pkl file
    os.makedirs("MLProject/models", exist_ok=True)
    joblib.dump(best_model, "MLProject/models/best_model.pkl")
    print("Best model saved as best_model.pkl")

    # Log the best model to MLflow
    mlflow.sklearn.log_model(best_model, "best_model")