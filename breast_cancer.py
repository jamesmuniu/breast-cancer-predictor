import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load and display the image with a specified width
image = Image.open('Capture.JPG')
st.image(image, width=80)

# Title and description
st.markdown("<h3 style='font-size: 20px;'>🔬🩺 Breast Cancer Prediction App. </h3>", unsafe_allow_html=True)

st.markdown("<h3 style='font-size: 20px;'>🔗Step by Step Guide👣👣👣</h3>", unsafe_allow_html=True)
(""" Download the Dataset""")
st.markdown("""Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17)
""")
("""
The dataset contains 📝:
569 cases : Each representing a unique patient record🗃️.
30 FNA-derived features : These features describe characteristics of the tumor, such as radius, texture, perimeter, area, smoothness, and more.
Prepare Your Data for Prediction
If you're testing the accuracy of the model:
Remove the target column (diagnosis) from your dataset before uploading. This column indicates whether each tumor is benign (0) or malignant (1).
Save the modified dataset as a CSV file .
 Upload Your CSV File
Use the upload feature on the left👈of the page to submit your prepared CSV file.
Ensure the file contains only the 30 feature columns without the target column. """)
st.markdown("<h3 style='font-size: 20px;'>🔗Run Predictions⚙️</h3>", unsafe_allow_html=True)
(""" Once uploaded, the pre-trained machine learning model will process your data and generate predictions.
The model will classify each tumor as either benign or malignant.""")
st.markdown("<h3 style='font-size: 20px;'>🔗Download Predictions⏬</h3>", unsafe_allow_html=True)
(""" After the predictions are complete, download the updated CSV file.
The downloaded file will include an additional column: Predictions , which contains the model's classification for each case.""")
st.markdown("<h3 style='font-size: 20px;'>🔗Compare Results☯️</h3>", unsafe_allow_html=True)
("""If you removed the target column earlier, you can now compare the Predictions column with the original diagnosis column to evaluate the model's accuracy.
""")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load model
@st.cache_resource
def load_model(model_path="trained_model_Neural_Network_(MLP).pkl"):
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)
    
    # Check if the loaded object is a dictionary and extract the model
    if isinstance(loaded_data, dict):
        st.sidebar.info("📁 Model file contains a dictionary. Looking for model...")
        
        # Try to find the model in the dictionary
        model_keys = [key for key in loaded_data.keys() if 'model' in key.lower() or 'classifier' in key.lower() or 'mlp' in key.lower()]
        
        if model_keys:
            model = loaded_data[model_keys[0]]
            st.sidebar.success(f"✅ Model found with key: '{model_keys[0]}'")
        else:
            # If no obvious model key, try to use the first value that has predict method
            for key, value in loaded_data.items():
                if hasattr(value, 'predict'):
                    model = value
                    st.sidebar.success(f"✅ Model found with key: '{key}'")
                    break
            else:
                # If no model found, show available keys and raise error
                st.sidebar.error(f"❌ No model found in dictionary. Available keys: {list(loaded_data.keys())}")
                raise ValueError("No model found in the dictionary")
    else:
        # If it's not a dictionary, assume it's the model directly
        model = loaded_data
    
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    model = None

# Perform prediction
if uploaded_file is not None and model is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(data.head())

        # Drop unwanted columns if present
        if "diagnosis" in data.columns:
            data_features = data.drop(columns=["diagnosis"])
        else:
            data_features = data.copy()

        # Make predictions (without scaling)
        st.info("🔄 Making predictions...")
        
        # Check if model has predict method
        if hasattr(model, 'predict'):
            predictions = model.predict(data_features)
            
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                prediction_probabilities = model.predict_proba(data_features)[:, 1]
            else:
                # If no predict_proba, create dummy probabilities
                prediction_probabilities = [0.5] * len(predictions)
                st.warning("⚠️ Model doesn't support probability predictions. Using default values.")
            
            # Add predictions to the dataframe
            data['Prediction'] = predictions
            data['Prediction Probability'] = prediction_probabilities

            # Map predictions to labels
            data['Prediction Label'] = data['Prediction'].map({0: 'Benign', 1: 'Malignant'})

            st.subheader("✅ Prediction Results")
            st.write(data[['Prediction Label', 'Prediction Probability']].head())

            # Show prediction distribution
            st.subheader("📊 Prediction Distribution")
            pred_counts = data['Prediction Label'].value_counts()
            st.write(pred_counts)

            # Option to download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            
        else:
            st.error("❌ The loaded object doesn't have a 'predict' method. Please check your model file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("💡 Tip: Make sure your CSV file has the same features as the training data.")
elif uploaded_file is not None and model is None:
    st.error("❌ Cannot make predictions - model failed to load.")
else:
    st.info("Please upload or drop a CSV file on the left👈to begin (for mobile phones there is a window slider button '>' on the top-left of the screen and remember, early detection saves lives.")

# Load and display the image with a specified width
image = Image.open('Capture.JPG')
st.image(image, width=150)
st.markdown("<h3 style='font-size: 20px;'>📢Breast Cancer Awareness.</h3>", unsafe_allow_html=True)
