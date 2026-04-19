import os
import json
import yaml
import joblib
import subprocess
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def main():

    # Load configuration
    config = load_config()

    processed_dir = config["data"]["processed_dir"]
    version = config["data"]["current_version"]
    model_params = config["model_params"]

    train_file = os.path.join(processed_dir, f"{version}_train.csv")

    print("Loading training data...")
    df = pd.read_csv(train_file)

    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    print("Training RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"]
    )

    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"Training Accuracy: {accuracy:.4f}")


    # Save versioned model
    os.makedirs("models", exist_ok=True)

    model_filename = f"model_{version}.pkl"
    model_path = os.path.join("models", model_filename)

    joblib.dump(model, model_path)


    # Save versioned metadata JSON
    metadata_filename = f"{version}_metadata.json"
    metadata_path = os.path.join("models", metadata_filename)

    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by_script": "src/train.py",
        "dataset_version": version,
        "git_commit_hash": get_git_commit_hash(),
        "algorithm": model_params["algorithm"],
        "model_parameters": model_params,
        "model_path": model_path,
        "training_accuracy": accuracy
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Append to model registry log
    log_entry = f"""
Model Version: {version}
Model File: {model_path}
Metadata File: {metadata_path}
Training Accuracy: {accuracy:.4f}
Git Commit: {metadata['git_commit_hash']}
Training Date: {metadata['training_date']}
------------------------------------------------------------
"""

    with open("models/model_metadata.log", "a") as log_file:
        log_file.write(log_entry)

    print("Model registry updated successfully.")


if __name__ == "__main__":
    main()
