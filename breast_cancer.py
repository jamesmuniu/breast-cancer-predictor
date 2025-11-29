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

# Load model and detect preprocessor type
@st.cache_resource
def load_model(model_path="trained_model_Neural_Network_(MLP).pkl"):
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)
    
    st.sidebar.info("🔍 Analyzing model structure...")
    
    # Check what type of object we loaded
    if isinstance(loaded_data, dict):
        st.sidebar.info("📁 Model file contains a dictionary")
        
        # Look for different possible components
        model = None
        pipeline = None
        scaler = None
        feature_names = None
        preprocessor = None
        
        # First, check if it's a pipeline (most common for preprocessed models)
        pipeline_keys = [key for key in loaded_data.keys() if 'pipeline' in key.lower()]
        if pipeline_keys:
            pipeline = loaded_data[pipeline_keys[0]]
            st.sidebar.success(f"✅ Found pipeline with key: '{pipeline_keys[0]}'")
        
        # Check for standalone model
        model_keys = [key for key in loaded_data.keys() if 'model' in key.lower() and 'pipeline' not in key.lower()]
        if model_keys and not pipeline:
            model = loaded_data[model_keys[0]]
            st.sidebar.success(f"✅ Found model with key: '{model_keys[0]}'")
        
        # Check for preprocessor
        preprocessor_keys = [key for key in loaded_data.keys() if 'preprocessor' in key.lower() or 'processor' in key.lower()]
        if preprocessor_keys:
            preprocessor = loaded_data[preprocessor_keys[0]]
            st.sidebar.success(f"✅ Found preprocessor with key: '{preprocessor_keys[0]}'")
        
        # Check for scaler
        scaler_keys = [key for key in loaded_data.keys() if 'scaler' in key.lower()]
        if scaler_keys:
            scaler = loaded_data[scaler_keys[0]]
            st.sidebar.success(f"✅ Found scaler with key: '{scaler_keys[0]}'")
        
        # Check for feature names
        feature_keys = [key for key in loaded_data.keys() if 'feature' in key.lower() or 'columns' in key.lower()]
        if feature_keys:
            feature_names = loaded_data[feature_keys[0]]
            st.sidebar.success(f"✅ Found feature names with key: '{feature_keys[0]}'")
        
        return {
            'model': model,
            'pipeline': pipeline,
            'scaler': scaler,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'raw_data': loaded_data
        }
    
    else:
        # Single object - check if it's a pipeline, model, or preprocessor
        st.sidebar.info("📁 Model file contains a single object")
        
        # Check if it's a pipeline
        if hasattr(loaded_data, 'steps') or hasattr(loaded_data, 'named_steps'):
            st.sidebar.success("✅ Loaded object is a Pipeline")
            return {
                'pipeline': loaded_data,
                'model': None,
                'scaler': None,
                'preprocessor': None,
                'feature_names': None,
                'raw_data': loaded_data
            }
        # Check if it's a model with predict method
        elif hasattr(loaded_data, 'predict'):
            st.sidebar.success("✅ Loaded object is a Model")
            return {
                'model': loaded_data,
                'pipeline': None,
                'scaler': None,
                'preprocessor': None,
                'feature_names': None,
                'raw_data': loaded_data
            }
        # Check if it's a preprocessor/scaler
        elif hasattr(loaded_data, 'transform'):
            st.sidebar.success("✅ Loaded object is a Preprocessor/Scaler")
            return {
                'model': None,
                'pipeline': None,
                'scaler': loaded_data,
                'preprocessor': loaded_data,
                'feature_names': None,
                'raw_data': loaded_data
            }
        else:
            st.sidebar.warning("⚠️ Unknown object type loaded")
            return {
                'model': None,
                'pipeline': None,
                'scaler': None,
                'preprocessor': None,
                'feature_names': None,
                'raw_data': loaded_data
            }

def detect_preprocessor_type(components):
    """Detect what type of preprocessor we have"""
    if components['pipeline'] is not None:
        return 'pipeline'
    elif components['preprocessor'] is not None:
        return 'preprocessor'
    elif components['scaler'] is not None:
        return 'scaler'
    else:
        return 'none'

try:
    components = load_model()
    preprocessor_type = detect_preprocessor_type(components)
    st.sidebar.success(f"✅ Model loaded! Preprocessor type: {preprocessor_type}")
    
    # Show what we found
    if components['pipeline']:
        st.sidebar.info(f"🔧 Pipeline steps: {len(components['pipeline'].steps) if hasattr(components['pipeline'], 'steps') else 'Unknown'}")
    if components['feature_names'] is not None:
        st.sidebar.info(f"📋 Expected features: {len(components['feature_names'])}")
        
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    components = None
    preprocessor_type = 'none'

def prepare_features_for_prediction(data_features, components, preprocessor_type):
    """
    Prepare features based on the type of preprocessor available
    """
    st.info(f"🔧 Preparing features for {preprocessor_type} preprocessor...")
    
    data_processed = data_features.copy()
    
    # Handle different preprocessor types
    if preprocessor_type == 'pipeline':
        st.write("🎯 Using full pipeline for preprocessing + prediction")
        # Pipeline handles everything internally - just return raw features
        # But we should validate feature names if available
        if components['feature_names'] is not None:
            expected_features = components['feature_names']
            missing_features = set(expected_features) - set(data_processed.columns)
            if missing_features:
                st.error(f"❌ Missing features for pipeline: {list(missing_features)}")
                return None
            data_processed = data_processed[expected_features]
        
    elif preprocessor_type == 'preprocessor':
        st.write("🔧 Applying standalone preprocessor")
        try:
            data_processed = components['preprocessor'].transform(data_processed)
            # If it returns numpy array, convert back to DataFrame if possible
            if isinstance(data_processed, np.ndarray) and components['feature_names'] is not None:
                data_processed = pd.DataFrame(data_processed, columns=components['feature_names'])
        except Exception as e:
            st.error(f"❌ Preprocessor transformation failed: {e}")
            return None
            
    elif preprocessor_type == 'scaler':
        st.write("🔧 Applying scaler")
        try:
            # Validate features if feature names available
            if components['feature_names'] is not None:
                expected_features = components['feature_names']
                missing_features = set(expected_features) - set(data_processed.columns)
                if missing_features:
                    st.error(f"❌ Missing features for scaler: {list(missing_features)}")
                    return None
                data_processed = data_processed[expected_features]
            
            data_processed = components['scaler'].transform(data_processed)
            if isinstance(data_processed, np.ndarray) and components['feature_names'] is not None:
                data_processed = pd.DataFrame(data_processed, columns=components['feature_names'])
        except Exception as e:
            st.error(f"❌ Scaler transformation failed: {e}")
            return None
            
    else:  # no preprocessor
        st.write("🔧 No preprocessor - using raw features")
        # Still validate feature names if available
        if components['feature_names'] is not None:
            expected_features = components['feature_names']
            missing_features = set(expected_features) - set(data_processed.columns)
            if missing_features:
                st.error(f"❌ Missing features: {list(missing_features)}")
                return None
            data_processed = data_processed[expected_features]
        
        # Convert all to numeric
        for col in data_processed.columns:
            if data_processed[col].dtype == 'object':
                data_processed[col] = pd.to_numeric(data_processed[col], errors='coerce')
                if data_processed[col].isna().any():
                    st.error(f"❌ Column '{col}' has non-convertible values")
                    return None
    
    st.success("✅ Feature preparation complete!")
    return data_processed

def make_predictions(data_processed, components, preprocessor_type):
    """
    Make predictions based on the available components
    """
    if preprocessor_type == 'pipeline':
        # Pipeline handles everything
        predictions = components['pipeline'].predict(data_processed)
        if hasattr(components['pipeline'], 'predict_proba'):
            probabilities = components['pipeline'].predict_proba(data_processed)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]
            else:
                probabilities = np.max(probabilities, axis=1)
        else:
            probabilities = np.ones(len(predictions)) * 0.5
            
    elif components['model'] is not None:
        # Use standalone model
        predictions = components['model'].predict(data_processed)
        if hasattr(components['model'], 'predict_proba'):
            probabilities = components['model'].predict_proba(data_processed)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]
            else:
                probabilities = np.max(probabilities, axis=1)
        else:
            probabilities = np.ones(len(predictions)) * 0.5
    else:
        st.error("❌ No model found for predictions")
        return None, None
        
    return predictions, probabilities

def check_data_quality(data):
    """Check for potential data quality issues"""
    warnings = []
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        warnings.append(f"⚠️ Found {missing_count} missing values")
    for col in data.columns:
        if data[col].nunique() == 1:
            warnings.append(f"⚠️ Column '{col}' has only one unique value")
    return warnings

# Perform prediction
if uploaded_file is not None and components is not None:
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
        
        # Prepare features - remove target column if present
        target_columns = ['diagnosis', 'satisfaction', 'target']
        found_targets = [col for col in target_columns if col in data.columns]
        
        if found_targets:
            data_features = data.drop(columns=found_targets)
            st.info(f"🗑️ Removed target column(s): {found_targets}")
        else:
            data_features = data.copy()
        
        # Prepare features based on preprocessor type
        st.subheader("🔧 Feature Preparation")
        data_processed = prepare_features_for_prediction(data_features, components, preprocessor_type)
        
        if data_processed is None:
            st.error("❌ Feature preparation failed")
            st.stop()
        
        st.success(f"✅ Features prepared! Shape: {data_processed.shape}")
        
        # Make predictions
        st.subheader("🎯 Making Predictions")
        predictions, probabilities = make_predictions(data_processed, components, preprocessor_type)
        
        if predictions is not None:
            # Add results to original data
            data['Prediction'] = predictions
            data['Prediction_Probability'] = probabilities
            
            # Map predictions to labels
            unique_preds = np.unique(predictions)
            if len(unique_preds) == 2 and set(unique_preds) == {0, 1}:
                data['Prediction_Label'] = data['Prediction'].map({0: 'Benign', 1: 'Malignant'})
            else:
                data['Prediction_Label'] = data['Prediction']
            
            # Show results
            st.subheader("✅ Prediction Results")
            st.write(data[['Prediction_Label', 'Prediction_Probability']].head())
            
            # Show distribution
            st.subheader("📊 Prediction Distribution")
            pred_counts = data['Prediction_Label'].value_counts()
            st.write(pred_counts)
            
            # Download option
            csv = data.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        
elif uploaded_file is not None and components is None:
    st.error("❌ Model not loaded properly")
else:
    st.info("👈 Please upload a CSV file to get started")

# Footer
image = Image.open('Capture.JPG')
st.image(image, width=150)
st.markdown("<h3 style='font-size: 20px;'>📢 Breast Cancer Awareness</h3>", unsafe_allow_html=True)
