import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Initialize components variable
components = None
preprocessor_type = 'none'

# Load and display the image with a specified width
#image = Image.open('Capture.JPG')
#st.image(image, width=80)

# Title and description
st.markdown("<h3 style='font-size: 20px;'>InsightNav AI ML Model Tester. </h3>", unsafe_allow_html=True)

st.markdown("<h3 style='font-size: 20px;'>🔗Step by Step Guide👣👣👣</h3>", unsafe_allow_html=True)
st.markdown("""Download the dataset""")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load model and detect preprocessor type
@st.cache_resource
def load_model(model_path="loan-approval_model_Neural_Network_(MLP) (1).pkl"):
    try:
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
            feature_keys = [key for key in loaded_data.keys() if 'feature' in key.lower() or 'columns' in key.lower() or 'features' in key.lower()]
            if feature_keys:
                feature_names = loaded_data[feature_keys[0]]
                st.sidebar.success(f"✅ Found feature names with key: '{feature_keys[0]}'")
                st.sidebar.info(f"📋 Model expects {len(feature_names)} features")
            
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
                # Try to extract feature names from pipeline
                feature_names = None
                try:
                    # For ColumnTransformer in pipeline
                    if hasattr(loaded_data, 'named_steps'):
                        for step_name, step in loaded_data.named_steps.items():
                            if hasattr(step, 'get_feature_names_out'):
                                feature_names = step.get_feature_names_out()
                                break
                except:
                    pass
                    
                return {
                    'pipeline': loaded_data,
                    'model': None,
                    'scaler': None,
                    'preprocessor': None,
                    'feature_names': feature_names,
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
                # Try to get feature names from preprocessor
                feature_names = None
                try:
                    if hasattr(loaded_data, 'get_feature_names_out'):
                        feature_names = loaded_data.get_feature_names_out()
                except:
                    pass
                return {
                    'model': None,
                    'pipeline': None,
                    'scaler': loaded_data,
                    'preprocessor': loaded_data,
                    'feature_names': feature_names,
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
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

def detect_preprocessor_type(_components):
    """Detect what type of preprocessor we have"""
    if _components is None:
        return 'none'
    if _components['pipeline'] is not None:
        return 'pipeline'
    elif _components['preprocessor'] is not None:
        return 'preprocessor'
    elif _components['scaler'] is not None:
        return 'scaler'
    else:
        return 'none'

def align_features_with_model(data_features, _components, _preprocessor_type):
    """
    Align the uploaded features with what the model expects
    """
    st.info("🔍 Aligning features with model expectations...")
    
    # Get expected features
    expected_features = _components['feature_names']
    
    if expected_features is None:
        st.error("❌ Cannot align features - no feature names found in model")
        st.info("💡 The model file should include the expected feature names")
        return None
    
    st.write(f"📋 Model expects {len(expected_features)} features: {list(expected_features)}")
    st.write(f"📊 Uploaded data has {len(data_features.columns)} features: {list(data_features.columns)}")
    
    # Find matching features
    uploaded_features = set(data_features.columns)
    expected_features_set = set(expected_features)
    
    # Find matches, missing, and extra features
    matching_features = uploaded_features.intersection(expected_features_set)
    missing_features = expected_features_set - uploaded_features
    extra_features = uploaded_features - expected_features_set
    
    st.write(f"✅ Matching features: {len(matching_features)}")
    
    if missing_features:
        st.error(f"❌ Missing features ({len(missing_features)}): {list(missing_features)}")
        return None
    
    if extra_features:
        st.warning(f"⚠️ Extra features ({len(extra_features)}) will be ignored: {list(extra_features)}")
    
    # Select only the expected features in the correct order
    data_aligned = data_features[expected_features].copy()
    
    st.success(f"✅ Features aligned! Using {len(data_aligned.columns)} features")
    return data_aligned

def prepare_features_for_prediction(data_features, _components, _preprocessor_type):
    """
    Prepare features based on the type of preprocessor available
    """
    st.info(f"🔧 Preparing features for {_preprocessor_type} preprocessor...")
    
    # First align features with model expectations
    data_aligned = align_features_with_model(data_features, _components, _preprocessor_type)
    if data_aligned is None:
        return None
    
    # Handle different preprocessor types
    if _preprocessor_type == 'pipeline':
        st.write("🎯 Using full pipeline for preprocessing + prediction")
        # Pipeline handles everything internally - just return aligned features
        return data_aligned
        
    elif _preprocessor_type == 'preprocessor':
        st.write("🔧 Applying standalone preprocessor")
        try:
            data_processed = _components['preprocessor'].transform(data_aligned)
            st.success("✅ Preprocessor transformation successful")
            
            # Don't try to convert back to DataFrame - keep as numpy array
            # The preprocessor output might have different feature names/numbers
            # due to one-hot encoding, feature selection, etc.
            st.write(f"📊 Preprocessor output shape: {data_processed.shape}")
            
            # If it's already a DataFrame, keep it as is
            if isinstance(data_processed, pd.DataFrame):
                return data_processed
            else:
                # For numpy arrays, we don't need column names for prediction
                # The model will work with the numpy array directly
                return data_processed
                
        except Exception as e:
            st.error(f"❌ Preprocessor transformation failed: {e}")
            return None
            
    elif _preprocessor_type == 'scaler':
        st.write("🔧 Applying scaler")
        try:
            data_processed = _components['scaler'].transform(data_aligned)
            st.success("✅ Scaler transformation successful")
            st.write(f"📊 Scaler output shape: {data_processed.shape}")
            
            if isinstance(data_processed, pd.DataFrame):
                return data_processed
            else:
                return data_processed
                
        except Exception as e:
            st.error(f"❌ Scaler transformation failed: {e}")
            return None
            
    else:  # no preprocessor
        st.write("🔧 No preprocessor - using raw features")
        # Convert all to numeric
        for col in data_aligned.columns:
            if data_aligned[col].dtype == 'object':
                data_aligned[col] = pd.to_numeric(data_aligned[col], errors='coerce')
                if data_aligned[col].isna().any():
                    st.error(f"❌ Column '{col}' has non-convertible values")
                    return None
        return data_aligned

def make_predictions(data_processed, _components, _preprocessor_type):
    """
    Make predictions based on the available components
    """
    st.write(f"📊 Making predictions on data with shape: {data_processed.shape}")
    
    if _preprocessor_type == 'pipeline':
        # Pipeline handles everything
        st.write("🚀 Making predictions with pipeline...")
        predictions = _components['pipeline'].predict(data_processed)
        if hasattr(_components['pipeline'], 'predict_proba'):
            probabilities = _components['pipeline'].predict_proba(data_processed)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]
            else:
                probabilities = np.max(probabilities, axis=1)
        else:
            probabilities = np.ones(len(predictions)) * 0.5
            
    elif _components['model'] is not None:
        # Use standalone model
        st.write("🚀 Making predictions with model...")
        predictions = _components['model'].predict(data_processed)
        if hasattr(_components['model'], 'predict_proba'):
            probabilities = _components['model'].predict_proba(data_processed)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]
            else:
                probabilities = np.max(probabilities, axis=1)
        else:
            probabilities = np.ones(len(predictions)) * 0.5
    else:
        st.error("❌ No model found for predictions")
        return None, None
        
    st.success(f"✅ Predictions completed! Generated {len(predictions)} predictions")
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

# Load the model when the app starts
if components is None:
    try:
        components = load_model()
        if components is not None:
            preprocessor_type = detect_preprocessor_type(components)
            st.sidebar.success(f"✅ Model loaded successfully! Preprocessor type: {preprocessor_type}")
        else:
            st.sidebar.error("❌ Failed to load model")
    except Exception as e:
        st.sidebar.error(f"❌ Error during model loading: {e}")
        components = None

# Perform prediction
if uploaded_file is not None:
    if components is not None:
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
                st.info("📋 Using all columns as features")
            
            # Show feature comparison
            st.subheader("🔍 Feature Comparison")
            if components['feature_names'] is not None:
                st.write(f"Model expects: {len(components['feature_names'])} features")
                st.write(f"Uploaded data has: {len(data_features.columns)} features")
            
            # Prepare features based on preprocessor type
            st.subheader("🔧 Feature Preparation")
            data_processed = prepare_features_for_prediction(data_features, components, preprocessor_type)
            
            if data_processed is None:
                st.error("❌ Feature preparation failed")
                st.info("💡 Possible solutions:")
                st.write("1. Ensure your CSV has the same features as the training data")
                st.write("2. Check that feature names match exactly (case-sensitive)")
                st.write("3. Verify no extra columns are present")
                st.write("4. Make sure all required features are present")
                st.stop()
            
            st.success(f"✅ Features prepared! Final shape: {data_processed.shape}")
            
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
                    data['Prediction_Label'] = data['Prediction'].map({0: 'Not_Approved', 1: 'Approved'})
                else:
                    data['Prediction_Label'] = data['Prediction']
                
                # Show results
                st.subheader("✅ Prediction Results")
                st.write(data[['Prediction_Label', 'Prediction_Probability']].head())
                
                # Show distribution
                st.subheader("📊 Prediction Distribution")
                pred_counts = data['Prediction_Label'].value_counts()
                st.write(pred_counts)
                
                # Show confidence statistics
                st.subheader("📈 Confidence Statistics")
                st.write(f"Average confidence: {data['Prediction_Probability'].mean():.3f}")
                st.write(f"Confidence range: {data['Prediction_Probability'].min():.3f} - {data['Prediction_Probability'].max():.3f}")
                
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
    else:
        st.error("❌ Model not loaded properly - cannot make predictions")
else:
    st.info("👈 Please upload a CSV file to get started")

# Footer
#image = Image.open('Capture.JPG')
#st.image(image, width=150)
#st.markdown("<h3 style='font-size: 20px;'>📢 Breast Cancer Awareness</h3>", unsafe_allow_html=True)
