import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from sklearn.base import is_classifier, is_regressor

# Initialize components variable
components = None
preprocessor_type = 'none'
model_type = 'unknown'

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

def detect_model_type(_components):
    """Detect if model is classification or regression"""
    model = None
    if _components is None:
        return 'unknown'
    
    if _components['pipeline'] is not None:
        model = _components['pipeline']
    elif _components['model'] is not None:
        model = _components['model']
    
    if model is None:
        return 'unknown'
    
    try:
        # Check if it's a classifier
        if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
            return 'classification'
        # Check if it's a regressor
        elif hasattr(model, '_estimator_type'):
            if model._estimator_type == 'regressor':
                return 'regression'
            elif model._estimator_type == 'classifier':
                return 'classification'
    except:
        pass
    
    # Try sklearn's type checking
    try:
        if is_classifier(model):
            return 'classification'
        elif is_regressor(model):
            return 'regression'
    except:
        pass
    
    # Check based on method availability
    if hasattr(model, 'predict_proba'):
        return 'classification'
    elif hasattr(model, 'predict'):
        # Default to regression if we can't determine
        st.warning("⚠️ Could not determine model type. Defaulting to regression.")
        return 'regression'
    
    return 'unknown'

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
            st.write(f"📊 Preprocessor output shape: {data_processed.shape}")
            
            if isinstance(data_processed, pd.DataFrame):
                return data_processed
            else:
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

def make_predictions(data_processed, _components, _preprocessor_type, _model_type):
    """
    Make predictions based on model type
    """
    st.write(f"📊 Making {_model_type} predictions on data with shape: {data_processed.shape}")
    
    predictions = None
    confidence_scores = None
    
    try:
        if _preprocessor_type == 'pipeline':
            predictions = _components['pipeline'].predict(data_processed)
            if _model_type == 'classification' and hasattr(_components['pipeline'], 'predict_proba'):
                proba = _components['pipeline'].predict_proba(data_processed)
                if proba.shape[1] == 2:
                    confidence_scores = proba[:, 1]  # Positive class probability
                else:
                    confidence_scores = np.max(proba, axis=1)  # Max probability for multiclass
            elif _model_type == 'regression':
                # For regression, we might use prediction intervals or other metrics
                confidence_scores = np.ones_like(predictions)  # Placeholder
        
        elif _components['model'] is not None:
            predictions = _components['model'].predict(data_processed)
            if _model_type == 'classification' and hasattr(_components['model'], 'predict_proba'):
                proba = _components['model'].predict_proba(data_processed)
                if proba.shape[1] == 2:
                    confidence_scores = proba[:, 1]
                else:
                    confidence_scores = np.max(proba, axis=1)
            elif _model_type == 'regression':
                confidence_scores = np.ones_like(predictions)  # Placeholder
        else:
            st.error("❌ No model found for predictions")
            return None, None
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None
    
    st.success(f"✅ Predictions completed! Generated {len(predictions)} predictions")
    return predictions, confidence_scores

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
            model_type = detect_model_type(components)
            st.sidebar.success(f"✅ Model loaded successfully!")
            st.sidebar.info(f"📊 Preprocessor type: {preprocessor_type}")
            st.sidebar.info(f"🎯 Model type: {model_type}")
        else:
            st.sidebar.error("❌ Failed to load model")
    except Exception as e:
        st.sidebar.error(f"❌ Error during model loading: {e}")
        components = None

# Main app logic
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
            target_columns = ['diagnosis', 'satisfaction', 'target', 'label', 'target_variable', 'y']
            found_targets = [col for col in target_columns if col in data.columns]
            
            if found_targets:
                data_features = data.drop(columns=found_targets)
                st.info(f"🗑️ Removed potential target column(s): {found_targets}")
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
            predictions, confidence_scores = make_predictions(data_processed, components, preprocessor_type, model_type)
            
            # FIXED: This condition was causing the error
            if predictions is not None and confidence_scores is not None:
                # Add results to original data
                data['Prediction'] = predictions
                
                if model_type == 'classification':
                    data['Prediction_Probability'] = confidence_scores
                    
                    # Map predictions to labels
                    unique_preds = np.unique(predictions)
                    
                    # Try to get class names from model
                    model_for_classes = None
                    if components['pipeline'] is not None:
                        model_for_classes = components['pipeline']
                    elif components['model'] is not None:
                        model_for_classes = components['model']
                    
                    if hasattr(model_for_classes, 'classes_') and model_for_classes.classes_ is not None:
                        # Create mapping from model's classes
                        class_mapping = {i: str(model_for_classes.classes_[i]) for i in range(len(model_for_classes.classes_))}
                        data['Prediction_Label'] = data['Prediction'].map(class_mapping)
                    else:
                        # Generic labels
                        if len(unique_preds) == 2 and set(unique_preds) == {0, 1}:
                            data['Prediction_Label'] = data['Prediction'].map({0: 'Class_0', 1: 'Class_1'})
                        else:
                            data['Prediction_Label'] = data['Prediction'].apply(lambda x: f'Class_{x}')
                    
                    # Show classification results
                    st.subheader("✅ Classification Results")
                    st.write(data[['Prediction_Label', 'Prediction_Probability']].head())
                    
                    # Show distribution
                    st.subheader("📊 Prediction Distribution")
                    pred_counts = data['Prediction_Label'].value_counts()
                    st.write(pred_counts)
                    
                    # Show confidence statistics
                    st.subheader("📈 Confidence Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Confidence", f"{data['Prediction_Probability'].mean():.3f}")
                    with col2:
                        st.metric("Min Confidence", f"{data['Prediction_Probability'].min():.3f}")
                    with col3:
                        st.metric("Max Confidence", f"{data['Prediction_Probability'].max():.3f}")
                    
                    # Create classification plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Bar plot for class distribution
                    pred_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
                    ax1.set_title('Class Distribution')
                    ax1.set_xlabel('Class')
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Histogram for confidence scores
                    ax2.hist(data['Prediction_Probability'], bins=20, edgecolor='black', alpha=0.7)
                    ax2.set_title('Confidence Score Distribution')
                    ax2.set_xlabel('Confidence Score')
                    ax2.set_ylabel('Frequency')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:  # Regression
                    # Add confidence scores if available
                    if confidence_scores is not None and len(confidence_scores) == len(predictions):
                        data['Confidence'] = confidence_scores
                    
                    # Show regression results
                    st.subheader("✅ Regression Results")
                    st.write(data[['Prediction']].head())
                    
                    # Show statistics
                    st.subheader("📊 Regression Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Prediction", f"{predictions.mean():.3f}")
                    with col2:
                        st.metric("Min Prediction", f"{predictions.min():.3f}")
                    with col3:
                        st.metric("Max Prediction", f"{predictions.max():.3f}")
                    
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Std Dev", f"{predictions.std():.3f}")
                    with col5:
                        st.metric("Median", f"{np.median(predictions):.3f}")
                    with col6:
                        st.metric("Range", f"{predictions.max() - predictions.min():.3f}")
                    
                    # Create regression plots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Histogram of predictions
                    ax1.hist(predictions, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
                    ax1.set_title('Distribution of Predictions')
                    ax1.set_xlabel('Predicted Value')
                    ax1.set_ylabel('Frequency')
                    ax1.grid(True, alpha=0.3)
                    
                    # Box plot
                    ax2.boxplot(predictions, vert=False)
                    ax2.set_title('Prediction Summary')
                    ax2.set_xlabel('Predicted Value')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Common download section for both types
                st.subheader("📥 Download Results")
                csv = data.to_csv(index=False)
                st.download_button(
                    "Download Predictions as CSV",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            else:
                st.error("❌ Failed to generate predictions")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    else:
        st.error("❌ Model not loaded properly - cannot make predictions")
else:
    st.info("👈 Please upload a CSV file to get started")

# Footer
#image = Image.open('Capture.JPG')
#st.image(image, width=150)
#st.markdown("<h3 style='font-size: 20px;'>📢 Breast Cancer Awareness</h3>", unsafe_allow_html=True)
