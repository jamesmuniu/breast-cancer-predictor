import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

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

def preprocess_data(data):
    """
    Preprocess the data by converting categorical strings to numerical values
    and ensuring all data is in the correct format for the model.
    """
    data_processed = data.copy()
    
    # Track which columns we converted
    converted_columns = {}
    
    for column in data_processed.columns:
        # Check if the column contains string data
        if data_processed[column].dtype == 'object':
            st.info(f"🔧 Converting categorical column: '{column}'")
            
            # Get unique values to show what we're converting
            unique_vals = data_processed[column].unique()
            st.write(f"   Unique values in '{column}': {list(unique_vals)}")
            
            # Convert categorical data to numerical
            # For binary categorical data (like Male/Female, Yes/No)
            if len(unique_vals) == 2:
                # Create mapping for binary categories
                mapping = {val: i for i, val in enumerate(unique_vals)}
                data_processed[column] = data_processed[column].map(mapping)
                converted_columns[column] = {'type': 'binary', 'mapping': mapping}
                st.write(f"   Binary mapping: {mapping}")
            
            # For multi-category data
            elif len(unique_vals) > 2:
                # Use one-hot encoding for multiple categories
                dummies = pd.get_dummies(data_processed[column], prefix=column)
                data_processed = pd.concat([data_processed, dummies], axis=1)
                data_processed = data_processed.drop(columns=[column])
                converted_columns[column] = {'type': 'one-hot', 'categories': list(unique_vals)}
                st.write(f"   One-hot encoded into {len(unique_vals)} columns")
            
            else:
                # Only one unique value - can be converted to 0 or dropped
                data_processed[column] = 0
                converted_columns[column] = {'type': 'constant', 'value': 0}
                st.write(f"   Constant value column, set to 0")
    
    return data_processed, converted_columns

def check_data_quality(data):
    """
    Check for potential data quality issues and provide warnings
    """
    warnings = []
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        warnings.append(f"⚠️ Found {missing_values} missing values in the data")
    
    # Check for infinite values
    if data.select_dtypes(include=[np.number]).size > 0:
        infinite_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            warnings.append(f"⚠️ Found {infinite_values} infinite values in the data")
    
    # Check for columns with all same values
    for col in data.columns:
        if data[col].nunique() == 1:
            warnings.append(f"⚠️ Column '{col}' has only one unique value")
    
    return warnings

# Perform prediction
if uploaded_file is not None and model is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write("Original data shape:", data.shape)
        st.write(data.head())

        # Show data types
        st.subheader("📊 Data Types Information")
        st.write(data.dtypes)

        # Check for data quality issues
        data_warnings = check_data_quality(data)
        if data_warnings:
            st.warning("Data Quality Warnings:")
            for warning in data_warnings:
                st.write(warning)

        # Drop unwanted columns if present
        if "diagnosis" in data.columns:
            data_features = data.drop(columns=["diagnosis"])
            st.info("🗑️ 'diagnosis' column removed for prediction")
        else:
            data_features = data.copy()

        # Preprocess the data
        st.subheader("🔧 Data Preprocessing")
        data_processed, conversion_info = preprocess_data(data_features)
        
        st.success("✅ Data preprocessing completed!")
        st.write("Processed data shape:", data_processed.shape)
        st.write("Processed data preview:")
        st.write(data_processed.head())

        # Make predictions
        st.info("🔄 Making predictions...")
        
        # Check if model has predict method
        if hasattr(model, 'predict'):
            predictions = model.predict(data_processed)
            
            # Check if model has predict_proba method
            if hasattr(model, 'predict_proba'):
                prediction_probabilities = model.predict_proba(data_processed)[:, 1]
            else:
                # If no predict_proba, create dummy probabilities
                prediction_probabilities = [0.5] * len(predictions)
                st.warning("⚠️ Model doesn't support probability predictions. Using default values.")
            
            # Add predictions to the original dataframe
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

            # Show conversion summary
            if conversion_info:
                st.subheader("🔧 Data Conversion Summary")
                for col, info in conversion_info.items():
                    if info['type'] == 'binary':
                        st.write(f"• '{col}': Binary mapping {info['mapping']}")
                    elif info['type'] == 'one-hot':
                        st.write(f"• '{col}': One-hot encoded into {len(info['categories'])} categories")
                    elif info['type'] == 'constant':
                        st.write(f"• '{col}': Constant value set to {info['value']}")

            # Option to download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            
        else:
            st.error("❌ The loaded object doesn't have a 'predict' method. Please check your model file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("💡 Tip: Make sure your CSV file has the correct format and compatible features.")
        # Show more detailed error information for debugging
        st.write("Debug info - please check:")
        st.write("1. Ensure all non-target columns are present")
        st.write("2. Check for mixed data types in columns")
        st.write("3. Verify there are no special characters in the data")
elif uploaded_file is not None and model is None:
    st.error("❌ Cannot make predictions - model failed to load.")
else:
    st.info("Please upload or drop a CSV file on the left👈to begin (for mobile phones there is a window slider button '>' on the top-left of the screen and remember, early detection saves lives.")

# Load and display the image with a specified width
image = Image.open('Capture.JPG')
st.image(image, width=150)
st.markdown("<h3 style='font-size: 20px;'>📢Breast Cancer Awareness.</h3>", unsafe_allow_html=True)
