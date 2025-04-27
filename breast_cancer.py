import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="🩺 Breast Cancer Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    /* General dark mode styling */
    body {
        background-color: #121212; /* Dark background */
        color: #ffffff; /* White text */
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e; /* Slightly lighter than main background */
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        color: #bb86fc; /* Lavender highlight for headers */
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e1e1e;
        border-radius: 5px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        color: #ffffff;
    }
    
    /* Matplotlib plots */
    .matplotlib-figure {
        background-color: #1e1e1e !important;
    }
</style>
""", unsafe_allow_html=True)

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
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Prediction Label', data=data, palette='Set2', ax=ax)
        ax.set_title("Count of Predicted Labels", fontsize=14, color="#ffffff")
        ax.set_xlabel("Prediction Label", fontsize=12, color="#ffffff")
        ax.set_ylabel("Count", fontsize=12, color="#ffffff")
        ax.tick_params(colors="#ffffff")  # Change tick colors
        st.pyplot(fig)

        # Option to download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            help="Download the predictions as a CSV file."
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# Load and display the image with a specified width
try:
    image = Image.open('Capture.JPG')
    st.image(image, caption='Breast Cancer Awareness', width=150)
except FileNotFoundError:
    st.warning("Image not found. Please ensure 'Capture.JPG' is in the correct directory.")
