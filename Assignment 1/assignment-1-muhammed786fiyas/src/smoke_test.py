import requests

BASE_URL = "http://127.0.0.1:8000/predict"

# Sample valid input
valid_payload = {
    "Air_temperature_K": 300,
    "Process_temperature_K": 310,
    "Rotational_speed_rpm": 1500,
    "Torque_Nm": 40,
    "Tool_wear_min": 5,
    "Type_L": 0,
    "Type_M": 1
}

def test_server_reachable():
    try:
        response = requests.post(BASE_URL, json=valid_payload)
        if response.status_code == 200:
            print("Test 1 Passed: Server reachable (Status 200)")
        else:
            print(f"Test 1 Failed: Unexpected status code {response.status_code}")
    except Exception as e:
        print(f"Test 1 Failed: Server not reachable ({e})")


def test_response_format():
    response = requests.post(BASE_URL, json=valid_payload)
    data = response.json()

    if "prediction" in data and "model_version" in data:
        print("Test 2 Passed: Correct response format")
    else:
        print("Test 2 Failed: Missing required fields in response")


def test_prediction_type():
    response = requests.post(BASE_URL, json=valid_payload)
    data = response.json()

    if isinstance(data.get("prediction"), int):
        print("Test 3 Passed: Prediction type is integer")
    else:
        print("Test 3 Failed: Prediction is not integer")


if __name__ == "__main__":
    print("Running Smoke Tests...\n")
    test_server_reachable()
    test_response_format()
    test_prediction_type()
