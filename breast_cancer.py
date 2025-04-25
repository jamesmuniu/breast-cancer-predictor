import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="🩺 Breast Cancer Prediction App", layout="wide")

# Title and description
st.title("🩺 Breast Cancer Prediction App")
st.markdown("""
Upload your breast cancer dataset in CSV format to predict whether a tumor is benign or malignant using a pre-trained machine learning model.
""")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load model and scaler
@st.cache_resource
def load_model_scaler(model_path="model.pkl", scaler_path="scaler.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

# Perform prediction
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(data.head())

        # Drop unwanted columns if present
        if "diagnosis" in data.columns:
            data_features = data.drop(columns=["diagnosis"])
        else:
            data_features = data.copy()

        # Scale the data
        data_scaled = scaler.transform(data_features)

        # Make predictions
        predictions = model.predict(data_scaled)
        prediction_probabilities = model.predict_proba(data_scaled)[:, 1]

        # Add predictions to the dataframe
        data['Prediction'] = predictions
        data['Prediction Probability'] = prediction_probabilities

        # Map predictions to labels
        data['Prediction Label'] = data['Prediction'].map({0: 'Benign', 1: 'Malignant'})

        st.subheader("✅ Prediction Results")
        st.write(data[['Prediction Label', 'Prediction Probability']])

        # Visualization: Count plot of predictions
        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots(figsize=(10, 10))  # Set width to 4 inches and height to 3 inches
        sns.countplot(x='Prediction Label', data=data, palette='Set2', ax=ax)
        ax.set_title("Count of Predicted Labels")
        st.pyplot(fig)

        # Option to download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")

from PIL import Image

# Load and display the image with a specified width
image = Image.open('Capture.jpg')
st.image(image, caption='Breast Cancer Awareness', width=300)