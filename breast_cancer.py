import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Add a background color */
    .stApp {
        background-color: #f4f4f9;
    }
    
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Style headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    
    /* Style buttons */
    div.stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    
    /* Style dataframes */
    .stDataFrame {
        background-color: #ecf0f1;
        border-radius: 5px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="🩺 Breast Cancer Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Breast Cancer Prediction App")
st.markdown("""
Upload your breast cancer dataset in CSV format to predict whether a tumor is benign or malignant using a pre-trained machine learning model.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file containing breast cancer data.")
    
    # Add a collapsible section for additional info
    with st.expander("ℹ️ About the App"):
        st.write("""
        This app uses a pre-trained machine learning model to predict whether a tumor is benign or malignant based on input features.
        """)

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

        # Display uploaded data preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(data.head())
        with col2:
            st.subheader("📊 Basic Statistics")
            st.write(data.describe())

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
        st.subheader("✅ Prediction Results")
        st.dataframe(data[['Prediction Label', 'Prediction Probability']])

        # Visualization: Count plot of predictions
        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Prediction Label', data=data, palette='Set2', ax=ax)
        ax.set_title("Count of Predicted Labels", fontsize=14)
        ax.set_xlabel("Prediction Label", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        # Option to download predictions
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

# Display image
try:
    image = Image.open('Capture.JPG')
    st.image(image, caption='Breast Cancer Awareness', use_column_width=True)
