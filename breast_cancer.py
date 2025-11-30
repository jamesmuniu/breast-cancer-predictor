import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('test_cleaned.csv')

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nTarget variable distribution:")
print(df['satisfaction'].value_counts())

# Data preprocessing
# Drop unnecessary columns
df_clean = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')

# Handle categorical variables
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

# Check for missing values
print("\nMissing values:")
print(df_encoded.isnull().sum())

# Prepare features and target
X = df_encoded.drop('satisfaction', axis=1)
y = df_encoded['satisfaction']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['neutral or dissatisfied', 'satisfied'],
           yticklabels=['neutral or dissatisfied', 'satisfied'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

def predict_satisfaction_with_confidence(sample_data):
    """
    Predict satisfaction for new sample data with explicit confidence scores
    """
    prediction = rf_model.predict(sample_data)[0]
    probabilities = rf_model.predict_proba(sample_data)[0]
    confidence_dict = dict(zip(rf_model.classes_, probabilities))
    
    # Get the confidence score for the predicted class
    confidence_score = confidence_dict[prediction]
    
    return prediction, confidence_score, confidence_dict

# Example: Make prediction on first test sample with confidence scores
sample_idx = 0
sample_data = X_test.iloc[sample_idx:sample_idx+1]
actual_label = y_test.iloc[sample_idx]

prediction, confidence, all_probabilities = predict_satisfaction_with_confidence(sample_data)

print(f"\nSample Prediction with Confidence Scores:")
print(f"Actual: {actual_label}")
print(f"Predicted: {prediction}")
print(f"Confidence Score: {confidence:.2%}")
print(f"All Probabilities: {all_probabilities}")

# Analyze confidence scores across all test predictions
y_pred_proba = rf_model.predict_proba(X_test)
confidence_scores = np.max(y_pred_proba, axis=1)  # Get the highest probability for each prediction

print(f"\nConfidence Statistics across all test predictions:")
print(f"Average Confidence: {np.mean(confidence_scores):.2%}")
print(f"Minimum Confidence: {np.min(confidence_scores):.2%}")
print(f"Maximum Confidence: {np.max(confidence_scores):.2%}")
print(f"Standard Deviation: {np.std(confidence_scores):.2%}")

# Check confidence distribution by actual class
test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'confidence': confidence_scores
})

print(f"\nConfidence by Actual Class:")
confidence_by_class = test_results.groupby('actual')['confidence'].agg(['mean', 'std', 'min', 'max'])
print(confidence_by_class)

# Plot confidence distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=test_results, x='confidence', hue='actual', bins=20, alpha=0.7)
plt.title('Distribution of Confidence Scores by Actual Class')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.axvline(x=0.5, color='red', linestyle='--', label='50% Confidence Threshold')
plt.legend()
plt.show()

# Save the model (optional)
import joblib
joblib.dump(rf_model, 'satisfaction_predictor.pkl')
print("\nModel saved as 'satisfaction_predictor.pkl'")
