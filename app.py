import streamlit as st
import pickle
import pandas as pd

# Load the trained Random Forest model
# Make sure 'random_forest_model.pkl' is in the same directory as app.py,
# or provide the correct path.
try:
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'random_forest_model.pkl' not found. Please make sure the model file is in the correct location.")
    st.stop() # Stop the app if the model file is not found

# --- Streamlit App Title and Description ---
st.title("Customer Churn Prediction App")
st.write("This app predicts whether a customer will churn based on their characteristics.")

# --- Input Fields for Features ---
# You will need to create input fields for each feature that your model expects.
# The input types (text_input, number_input, selectbox, etc.) should match the
# data types and expected values of your features.

# Example Input Fields (replace with your actual features)
st.sidebar.header("Customer Features")

gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1]) # Assuming 0 for No, 1 for Yes
partner = st.sidebar.selectbox("Partner", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['Yes', 'No'])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
phone_service = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.sidebar.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
online_backup = st.sidebar.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.sidebar.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.sidebar.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.sidebar.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=20.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=20.0)


# --- Prepare Input Data for Prediction ---
# The input data needs to be in the same format (columns and encoding)
# as the data used to train the model (X_train_encoded).

# Create a dictionary from the input values
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert the dictionary to a pandas DataFrame
input_df = pd.DataFrame([input_data])

# Apply the same preprocessing (one-hot encoding) as used during training
# This part needs to be adapted to your specific encoding steps.
# Make sure the columns match X_train_encoded.columns

# Identify categorical columns (excluding numerical ones)
categorical_cols = input_df.select_dtypes(include='object').columns.tolist()

# Apply one-hot encoding
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Ensure the input columns match the training columns. Add missing columns with value 0.
# You'll need the list of columns from X_train_encoded.
# For example: training_cols = ['SeniorCitizen', 'tenure', ...] # Replace with your actual columns
# This is a crucial step to avoid errors during prediction.
# For now, I'll use a placeholder. You should replace this with the actual columns
# from your X_train_encoded DataFrame.
# Example (replace with your actual columns from X_train_encoded):
# training_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
# input_encoded = input_encoded.reindex(columns=training_cols, fill_value=0)

# --- Prediction ---
if st.button("Predict Churn"):
    # Ensure the input_encoded DataFrame has the same columns as X_train_encoded
    # This is a critical step. You need to load or define the list of columns
    # that your model was trained on.
    # For demonstration purposes, let's assume you have a list of training columns
    # called `training_columns`.
    # If you don't have this list readily available, you might need to save it
    # during your model training process.

    # **IMPORTANT:** Replace the following line with the actual column alignment code
    # using the columns from your X_train_encoded DataFrame.
    # For example:
    # input_encoded = input_encoded.reindex(columns=rf_model.feature_names_in_, fill_value=0)
    # Or, if using the list from the previous notebook:
    # training_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    # input_encoded = input_encoded.reindex(columns=training_cols, fill_value=0)

    # --- Placeholder for actual column alignment ---
    # In a real application, you would align the columns here.
    # For now, I'll just display the encoded input for verification.
    st.write("Encoded Input Data (for verification):")
    st.dataframe(input_encoded)
    # --- End Placeholder ---


    # Assuming column alignment is handled correctly before this point
    # prediction = rf_model.predict(input_encoded)
    # prediction_proba = rf_model.predict_proba(input_encoded)[:, 1] # Probability of churning

    # --- Placeholder for prediction ---
    # Uncomment the prediction code above and remove the placeholder below
    # once you have implemented the correct column alignment.
    st.warning("Prediction is placeholder until column alignment is implemented.")
    prediction = ["Placeholder"] # Placeholder prediction
    prediction_proba = [0.0] # Placeholder probability
    # --- End Placeholder ---


    st.subheader("Prediction Result:")
    if prediction[0] == 'Yes':
        st.write("The model predicts that this customer **will churn**.")
    else:
        st.write("The model predicts that this customer **will not churn**.")

    st.write(f"Probability of churning: {prediction_proba[0]:.4f}")
