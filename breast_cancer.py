import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.base import is_classifier, is_regressor
import matplotlib.pyplot as plt

# Add model type detection
def detect_model_type(_components):
    """Detect if model is classification or regression"""
    model = None
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
    
    return 'unknown'

# Modify the prediction function
def make_predictions(data_processed, _components, _preprocessor_type, model_type):
    """
    Make predictions based on model type
    """
    st.write(f"📊 Making {model_type} predictions on data with shape: {data_processed.shape}")
    
    predictions = None
    confidence_scores = None
    
    try:
        if _preprocessor_type == 'pipeline':
            predictions = _components['pipeline'].predict(data_processed)
            if model_type == 'classification' and hasattr(_components['pipeline'], 'predict_proba'):
                proba = _components['pipeline'].predict_proba(data_processed)
                if proba.shape[1] == 2:
                    confidence_scores = proba[:, 1]  # Positive class probability
                else:
                    confidence_scores = np.max(proba, axis=1)  # Max probability for multiclass
            elif model_type == 'regression':
                # For regression, we might use prediction intervals or other metrics
                confidence_scores = np.ones_like(predictions)  # Placeholder
        
        elif _components['model'] is not None:
            predictions = _components['model'].predict(data_processed)
            if model_type == 'classification' and hasattr(_components['model'], 'predict_proba'):
                proba = _components['model'].predict_proba(data_processed)
                if proba.shape[1] == 2:
                    confidence_scores = proba[:, 1]
                else:
                    confidence_scores = np.max(proba, axis=1)
            elif model_type == 'regression':
                confidence_scores = np.ones_like(predictions)  # Placeholder
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None
    
    return predictions, confidence_scores

# Modify the results display section
if predictions is not None:
    # Add results to original data
    data['Prediction'] = predictions
    
    if model_type == 'classification':
        data['Prediction_Probability'] = confidence_scores
        
        # Map predictions to labels if binary classification
        unique_preds = np.unique(predictions)
        if len(unique_preds) == 2 and set(unique_preds) == {0, 1}:
            # Use generic labels or detect from model
            if hasattr(model, 'classes_') and len(model.classes_) == 2:
                data['Prediction_Label'] = data['Prediction'].map(
                    {0: f'Class_{model.classes_[0]}', 1: f'Class_{model.classes_[1]}'}
                )
            else:
                data['Prediction_Label'] = data['Prediction'].map({0: 'Class_0', 1: 'Class_1'})
        else:
            data['Prediction_Label'] = predictions
        
        # Classification-specific stats
        st.subheader("📊 Classification Results")
        st.write(f"Predicted classes: {unique_preds}")
        st.write(f"Class distribution: {data['Prediction_Label'].value_counts().to_dict()}")
        
    else:  # Regression
        data['Prediction'] = predictions
        
        # Regression-specific stats
        st.subheader("📊 Regression Results")
        st.write(f"Prediction range: {predictions.min():.3f} - {predictions.max():.3f}")
        st.write(f"Mean prediction: {predictions.mean():.3f}")
        st.write(f"Std of predictions: {predictions.std():.3f}")
        
        # Create histogram of predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predictions')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Common statistics for both types
    st.subheader("📈 Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Predictions", len(predictions))
    with col2:
        st.metric("Unique Values", len(np.unique(predictions)))
    with col3:
        if model_type == 'classification':
            st.metric("Most Frequent Class", data['Prediction'].mode()[0])
        else:
            st.metric("Median Prediction", np.median(predictions))
    
    # Download option
    csv = data.to_csv(index=False)
    st.download_button(
        "📥 Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
