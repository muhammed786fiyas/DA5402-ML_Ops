import requests
import pandas as pd
import json
import yaml

BASE_URL = "http://127.0.0.1:8000/predict"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    version = config["data"]["current_version"]

    minimum_required_logs = config["monitoring"]["minimum_required_logs"]
    drift_threshold = config["monitoring"]["drift_threshold"]

    # Load production dataset
    production_file = f"data/production/{version}_production.csv"
    df = pd.read_csv(production_file)

    X = df.drop("Machine failure", axis=1)
    y_true = df["Machine failure"]

    print("Sending production data through API...")

    for _, row in X.iterrows():
        payload = {
            "Air_temperature_K": row["Air temperature [K]"],
            "Process_temperature_K": row["Process temperature [K]"],
            "Rotational_speed_rpm": row["Rotational speed [rpm]"],
            "Torque_Nm": row["Torque [Nm]"],
            "Tool_wear_min": row["Tool wear [min]"],
            "Type_L": row.get("Type_L", 0),
            "Type_M": row.get("Type_M", 0)
        }

        requests.post(BASE_URL, json=payload)

    # Read API logs
    api_log_file = "api_predictions_log.csv"
    api_logs = pd.read_csv(api_log_file)

    if len(api_logs) < minimum_required_logs:
        print(f"\nNot enough production data yet.")
        print(f"Current logs: {len(api_logs)}")
        print(f"Minimum required: {minimum_required_logs}")
        return

    recent_logs = api_logs.tail(len(y_true))

    predictions = recent_logs["prediction"].reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)

    production_accuracy = (predictions == y_true).mean()
    production_error = 1 - production_accuracy

    print(f"\nProduction Accuracy: {production_accuracy:.4f}")
    print(f"Production Error Rate: {production_error:.4f}")

    # Load training accuracy
    metadata_file = f"models/{version}_metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    training_accuracy = metadata["training_accuracy"]
    training_error = 1 - training_accuracy

    print(f"\nTraining Accuracy: {training_accuracy:.4f}")
    print(f"Training Error Rate: {training_error:.4f}")

    # Drift detection
    if production_error > training_error + drift_threshold:
        print("\n⚠ Drift Detected! Retraining Required.")
    else:
        print("\n✔ Model Performance Stable.")

if __name__ == "__main__":
    main()
