import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
#st.set_page_config(page_title="🩺 Breast Cancer Prediction App", layout="wide")
# Load and display the image with a specified width
image = Image.open('Capture.JPG')
st.image(image, width=80)

# Title and description
#st.title("🩺 Breast Cancer Prediction App")
st.markdown("<h3 style='font-size: 20px;'>🔬🩺 Breast Cancer Prediction App. </h3>", unsafe_allow_html=True)


st.markdown("<h3 style='font-size: 20px;'>🔗Step by Step Guide👣👣👣</h3>", unsafe_allow_html=True)
(""" Download the Dataset""")
st.markdown("""Download the dataset from the [UCI Machine Learning Repository(https://archive.ics.uci.edu/dataset/17)
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

# Load model and scaler
@st.cache_resource
def load_model_scaler(model_path="trained_model_Neural_Network_(MLP).pkl""):
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
        #st.subheader("📊 Prediction Distribution")
        #fig, ax = plt.subplots(figsize=(10, 10))  # Set width to 4 inches and height to 3 inches
        #sns.countplot(x='Prediction Label', data=data, palette='Set2', ax=ax)
        #ax.set_title("Count of Predicted Labels")
        #st.pyplot(fig)

        # Option to download
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload or drop a CSV file on the left👈to begin (for mobile phones there is a window slider button '>' on the top-left of the screen and remember, early detection saves lives.")

from PIL import Image

# Load and display the image with a specified width
image = Image.open('Capture.JPG')
st.image(image, width=150)
st.markdown("<h3 style='font-size: 20px;'>📢Breast Cancer Awareness.</h3>", unsafe_allow_html=True)


