import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Load the dataset
data = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")


# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split into train and test sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)


# Train the Logistic Regression model
model = LogisticRegression(max_iter=6000)
model.fit(X_train, y_train)


# Predict on test data
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)



# Accuracy
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# --- Streamlit Web App ---

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")

# Theme Toggle
theme = st.sidebar.radio("Choose Theme Mode", ("Dark", "Light"))

if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
            color: skyBlue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Add background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1604079628041-9434cd163b33?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
background-size: cover;
background-position: center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# App title
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("Predict whether a transaction is **Legitimate** or **Fraudulent** based on the input features.", unsafe_allow_html=True)

# Display model accuracy
with st.sidebar:
    st.header("Model Performance 📊")
    st.success(f"Training Accuracy: {train_acc*100:.2f}%")
    st.success(f"Testing Accuracy: {test_acc*100:.2f}%")
    st.markdown("---")
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=200)
    st.markdown("Now you can detect **Transactions**")


# Input section
st.subheader("🔎 Enter the Transaction Features")

with st.form(key='input_form'):
    input_data = st.text_area("Enter feature values separated by commas (,)", placeholder="Example: 0.1, 0.2, ..., -0.3")
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    try:
        features = np.asarray(input_data.split(','), dtype=np.float64)
        prediction = model.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction Detected!")
        else:
            st.error("⚠️ Fraudulent Transaction Detected!")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Please ensure you entered correct numeric values separated by commas.")



# --- Bulk Prediction from CSV Upload ---
st.subheader("📥 Upload a CSV file for Bulk Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        uploaded_data = pd.read_csv(uploaded_file)
        
        # Check and drop 'Class' column if it exists
        if 'Class' in uploaded_data.columns:
            uploaded_data = uploaded_data.drop(columns=['Class'])
            st.info("'Class' column was found in uploaded file and automatically removed.")

        st.write("Preview of Uploaded Data:")
        st.dataframe(uploaded_data.head())

        # Check if the uploaded data has the same number of features
        if uploaded_data.shape[1] != X_train.shape[1]:
            st.error(f"Uploaded data must have {X_train.shape[1]} features, but found {uploaded_data.shape[1]}. Please correct the file.")
        else:
            # Predict on the uploaded data
            bulk_predictions = model.predict(uploaded_data)
            
            # Add a prediction column
            uploaded_data['Prediction'] = ['Legitimate' if pred == 0 else 'Fraudulent' for pred in bulk_predictions]

            st.success("✅ Predictions completed!")
            st.write(uploaded_data)

            # Option to download the results
            csv = uploaded_data.to_csv(index=False)
            st.download_button("Download Predictions as CSV", data=csv, file_name='bulk_predictions.csv', mime='text/csv')

            


    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Make sure the CSV contains only the required features without the target column.")


# --- Model Evaluation ---
st.subheader("📈 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

with col2:
    st.markdown("#### ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)