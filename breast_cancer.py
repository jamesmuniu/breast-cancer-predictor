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
st.markdown("""Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17)""")

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
    """
    data_processed = data.copy()
    converted_columns = {}
    
    # First, identify all categorical columns
    categorical_columns = []
    for col in data_processed.columns:
        # Safely check if this is a categorical column
        try:
            if data_processed[col].dtype == 'object':
                categorical_columns.append(col)
        except AttributeError:
            # If we can't check dtype, assume it's not categorical
            continue
    
    st.info(f"🔧 Found {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Process each categorical column
    for col in categorical_columns:
        if col not in data_processed.columns:
            continue
            
        st.write(f"Processing column: '{col}'")
        
        # Get the actual series for this column
        col_series = data_processed[col]
        
        # Get unique values
        unique_vals = col_series.unique()
        st.write(f"   Unique values: {list(unique_vals)}")
        
        # Handle based on number of unique values
        if len(unique_vals) == 2:
            # Binary encoding
            mapping = {val: i for i, val in enumerate(unique_vals)}
            data_processed[col] = data_processed[col].map(mapping)
            converted_columns[col] = {'type': 'binary', 'mapping': mapping}
            st.write(f"   Binary mapping: {mapping}")
            
        elif len(unique_vals) > 2:
            # One-hot encoding
            dummies = pd.get_dummies(data_processed[col], prefix=col)
            data_processed = pd.concat([data_processed.drop(columns=[col]), dummies], axis=1)
            converted_columns[col] = {'type': 'one-hot', 'categories': list(unique_vals)}
            st.write(f"   One-hot encoded into {len(unique_vals)} columns")
            
        else:
            # Single value - convert to constant
            data_processed[col] = 0
            converted_columns[col] = {'type': 'constant', 'value': 0}
            st.write(f"   Constant value: 0")
    
    # Now handle any string columns that might have been missed
    for col in data_processed.columns:
        try:
            # Try to convert any remaining object columns to numeric
            if data_processed[col].dtype == 'object':
                original_dtype = data_processed[col].dtype
                data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
                if data_processed[col].isna().any():
                    st.warning(f"⚠️ Column '{col}' has non-convertible values")
                else:
                    st.info(f"🔧 Converted '{col}' from {original_dtype} to numeric")
                    converted_columns[col] = {'type': 'auto_numeric'}
        except:
            continue
    
    return data_processed, converted_columns

def check_data_quality(data):
    """
    Check for potential data quality issues
    """
    warnings = []
    
    # Check for missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        warnings.append(f"⚠️ Found {missing_count} missing values")
    
    # Check for constant columns
    for col in data.columns:
        if data[col].nunique() == 1:
            warnings.append(f"⚠️ Column '{col}' has only one unique value")
    
    return warnings

# Perform prediction
if uploaded_file is not None and model is not None:
    try:
        # Read the data
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(f"Data shape: {data.shape}")
        st.write(data.head())
        
        # Show data info
        st.subheader("📊 Data Information")
        st.write("Data types:")
        st.write(data.dtypes)
        
        # Check data quality
        warnings = check_data_quality(data)
        if warnings:
            st.warning("Data Quality Issues:")
            for warning in warnings:
                st.write(f"• {warning}")
        
        # Prepare features
        if "diagnosis" in data.columns:
            data_features = data.drop(columns=["diagnosis"])
            st.info("🗑️ Removed 'diagnosis' column for prediction")
        else:
            data_features = data.copy()
        
        # Preprocess data
        st.subheader("🔧 Data Preprocessing")
        data_processed, conversion_info = preprocess_data(data_features)
        
        st.success(f"✅ Preprocessing complete! Final shape: {data_processed.shape}")
        st.write("Processed data preview:")
        st.write(data_processed.head())
        st.write("Processed data types:")
        st.write(data_processed.dtypes)
        
        # Make predictions
        st.subheader("🎯 Making Predictions")
        
        if hasattr(model, 'predict'):
            # Check for any remaining non-numeric columns
            non_numeric_cols = []
            for col in data_processed.columns:
                if data_processed[col].dtype == 'object':
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                st.error(f"❌ The following columns are still non-numeric and cannot be processed: {non_numeric_cols}")
                st.stop()  # Use st.stop() instead of return
            else:
                # Make predictions
                predictions = model.predict(data_processed)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(data_processed)[:, 1]
                else:
                    probabilities = [0.5] * len(predictions)
                    st.warning("⚠️ Using default probabilities (0.5)")
                
                # Add results to original data
                data['Prediction'] = predictions
                data['Prediction_Probability'] = probabilities
                data['Prediction_Label'] = data['Prediction'].map({0: 'Benign', 1: 'Malignant'})
                
                # Show results
                st.subheader("✅ Prediction Results")
                st.write(data[['Prediction_Label', 'Prediction_Probability']].head())
                
                # Show distribution
                st.subheader("📊 Prediction Distribution")
                pred_counts = data['Prediction_Label'].value_counts()
                st.write(pred_counts)
                
                # Show conversion summary
                if conversion_info:
                    st.subheader("🔧 Conversion Summary")
                    for col, info in conversion_info.items():
                        if info['type'] == 'binary':
                            st.write(f"• **{col}**: Binary encoding {info['mapping']}")
                        elif info['type'] == 'one-hot':
                            st.write(f"• **{col}**: One-hot encoded into {len(info['categories'])} columns")
                        elif info['type'] == 'constant':
                            st.write(f"• **{col}**: Constant value {info['value']}")
                        elif info['type'] == 'auto_numeric':
                            st.write(f"• **{col}**: Auto-converted to numeric")
                
                # Download option
                csv = data.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "breast_cancer_predictions.csv",
                    "text/csv"
                )
            
        else:
            st.error("❌ Model doesn't have predict method")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Debugging tips:")
        st.write("1. Check that your CSV has the expected columns")
        st.write("2. Ensure all data is numeric or convertible to numeric")
        st.write("3. Verify the model expects the same features as your data")
        
elif uploaded_file is not None and model is None:
    st.error("❌ Model not loaded properly")
else:
    st.info("👈 Please upload a CSV file to get started")

# Footer
image = Image.open('Capture.JPG')
st.image(image, width=150)
st.markdown("<h3 style='font-size: 20px;'>📢 Breast Cancer Awareness</h3>", unsafe_allow_html=True)
