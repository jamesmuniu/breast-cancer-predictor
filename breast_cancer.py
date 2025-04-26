import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

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
        
        # Validate data
        if "diagnosis" not in data.columns:
            st.error("The uploaded dataset must contain a 'diagnosis' column or match the expected format.")
            st.stop()
        if not all(data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            st.error("The dataset must contain only numeric columns for prediction.")
            st.stop()

        st.subheader("📄 Uploaded Data Preview")
        st.write(data.head())

        # Drop unwanted columns if present
        data_features = data.drop(columns=["diagnosis"]) if "diagnosis" in data.columns else data.copy()

        # Scale the data
        data_scaled = scaler.transform(data_features)

        # Make predictions
        predictions = model.predict(data_scaled)
        prediction_probabilities = model.predict_proba(data_scaled)[:, 1]

        # Add predictions to the dataframe
        data['Prediction'] = predictions
        data['Prediction Probability'] = prediction_probabilities
        data['Prediction Label'] = data['Prediction'].map({0: 'Benign', 1: 'Malignant'})

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Uploaded Data Preview")
            st.write(data.head())
        with col2:
            st.subheader("✅ Prediction Results")
            st.write(data[['Prediction Label', 'Prediction Probability']])

        # Visualization: Count plot of predictions
        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
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

# Display image
try:
    image = Image.open('Capture.JPG')
    st.image(image, caption='Breast Cancer Awareness', width=150)
except FileNotFoundError:
    st.warning("Image not found. Please ensure 'Capture.JPG' is in the correct directory.")
