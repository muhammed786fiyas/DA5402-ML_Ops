import os
import requests
from flask import Flask, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MAILTRAP_API_TOKEN = os.getenv("MAILTRAP_API_TOKEN")
TO_EMAIL = "beacontech4@gmail.com"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    for alert in data.get("alerts", []):
        alertname   = alert["labels"].get("alertname", "Unknown")
        status      = alert.get("status", "unknown").upper()
        severity    = alert["labels"].get("severity", "unknown")
        summary     = alert["annotations"].get("summary", "No summary")
        description = alert["annotations"].get("description", "")

        response = requests.post(
            "https://sandbox.api.mailtrap.io/api/send/4480336",
            headers={
                "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "from": {"email": "alerts@mlops-a5.com", "name": "MLOps AlertManager"},
                "to": [{"email": TO_EMAIL}],
                "subject": f"[{status}] MLOps Alert: {alertname}",
                "html": f"""
                    <h2>{alertname}</h2>
                    <p><b>Status:</b> {status}</p>
                    <p><b>Severity:</b> {severity}</p>
                    <p><b>Summary:</b> {summary}</p>
                    <p><b>Description:</b> {description}</p>
                """
            }
        )
        print(f"Alert: {alertname} | Status: {response.status_code} | Response: {response.text}")
    return {"status": "ok"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)