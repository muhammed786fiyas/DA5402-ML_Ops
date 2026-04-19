import streamlit as st
import requests
import json
import torch
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import preprocess_image, tensor_to_list
from logger import get_logger

logger = get_logger("app")

# Config
MLFLOW_SERVING_URL = "http://127.0.0.1:5001/invocations"
TEMP_IMAGE_PATH    = "temp_upload.png"

# Page Setup
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Classifier")
st.markdown("Upload an image of a handwritten digit (0-9) and the model will classify it.")
st.divider()

# File Upload
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload a grayscale image of a handwritten digit"
)

if uploaded_file is not None:

    # Show uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, width=200)

    # Save temporarily for preprocessing
    image.save(TEMP_IMAGE_PATH)
    logger.info(f"Image uploaded: {uploaded_file.name}")

    # Preprocess & Call MLflow API
    with st.spinner("Classifying..."):
        try:
            tensor  = preprocess_image(TEMP_IMAGE_PATH)
            payload = {"inputs": tensor_to_list(tensor)}

            response = requests.post(
                MLFLOW_SERVING_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )

            if response.status_code == 200:
                logits = response.json()["predictions"][0]

                # Convert logits to probabilities
                logits_tensor   = torch.tensor(logits)
                probabilities   = torch.softmax(logits_tensor, dim=0).numpy()
                predicted_digit = int(np.argmax(probabilities))
                confidence      = float(probabilities[predicted_digit]) * 100

                logger.info(f"Prediction: {predicted_digit} | Confidence: {confidence:.2f}%")

                # Display Results
                with col2:
                    st.subheader("Prediction")
                    st.metric(
                        label="Predicted Digit",
                        value=str(predicted_digit)
                    )
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.2f}%"
                    )

                st.divider()
                st.subheader("Probability Distribution")

                # Bar chart of all probabilities
                prob_dict = {
                    str(i): float(probabilities[i]) * 100
                    for i in range(10)
                }
                st.bar_chart(prob_dict)

            else:
                logger.error(f"MLflow API error: {response.status_code}")
                st.error(f"MLflow API error: {response.status_code} — {response.json()}")

        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to MLflow serving server")
            st.error("Cannot connect to MLflow serving server at port 5001. Make sure it is running!")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error(f"Unexpected error: {e}")

        finally:
            # Clean up temp file
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)

st.divider()
st.caption("DA5402 MLOps Assignment 7 | MNIST Classifier | MLflow + Streamlit")