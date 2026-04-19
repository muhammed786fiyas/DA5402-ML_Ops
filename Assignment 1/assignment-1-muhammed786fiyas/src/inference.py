import os
import json
import yaml
import joblib
import subprocess
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel


# Load Config
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_git_commit_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode("utf-8").strip()

config = load_config()
version = config["data"]["current_version"]

model_path_template = config["deployment"]["model_path_template"]
model_path = model_path_template.format(version=version)

# Load Model
model = joblib.load(model_path)


# Deployment Logging
deployment_log_path = "deployment_log.csv"

if not os.path.exists(deployment_log_path):
    with open(deployment_log_path, "w") as f:
        f.write("timestamp,model_version,git_commit_hash\n")

with open(deployment_log_path, "a") as f:
    f.write(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
        f"{version},"
        f"{get_git_commit_hash()}\n"
    )

# FastAPI App
app = FastAPI()

class InputData(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Type_L: int
    Type_M: int

@app.post("/predict")
def predict(data: InputData):
    input_dict = {
        "Air temperature [K]": data.Air_temperature_K,
        "Process temperature [K]": data.Process_temperature_K,
        "Rotational speed [rpm]": data.Rotational_speed_rpm,
        "Torque [Nm]": data.Torque_Nm,
        "Tool wear [min]": data.Tool_wear_min,
        "Type_L": data.Type_L,
        "Type_M": data.Type_M
    }

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]


    # Log prediction to API log
    api_log_path = "api_predictions_log.csv"

    if not os.path.exists(api_log_path):
        with open(api_log_path, "w") as f:
            f.write("timestamp,model_version,prediction\n")

    with open(api_log_path, "a") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
            f"{version},"
            f"{int(prediction)}\n"
        )

    return {
        "model_version": version,
        "prediction": int(prediction)
    }
