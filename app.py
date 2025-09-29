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

# Identify categorical columns (excluding numerical ones)
categorical_cols = input_df.select_dtypes(include='object').columns.tolist()

# Apply one-hot encoding
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# --- Prediction ---
if st.button("Predict Churn"):
    try:
        # Get the list of feature names from the trained model
        # This is the correct way to get the columns the model expects
        training_cols = rf_model.feature_names_in_

        # Align the user's input data to match the model's training columns
        # This is a critical step to prevent prediction errors
        input_aligned = input_encoded.reindex(columns=training_cols, fill_value=0)

        # Make the prediction using the aligned data
        prediction = rf_model.predict(input_aligned)
        prediction_proba = rf_model.predict_proba(input_aligned)[:, 1] # Probability of churning

        # Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 'Yes':
            st.markdown("The model predicts that this customer **will churn**.")
        else:
            st.markdown("The model predicts that this customer **will not churn**.")

        st.write(f"Probability of churning: **{prediction_proba[0]:.4f}**")

    except AttributeError:
        st.error("The loaded model does not have a 'feature_names_in_' attribute. "
                 "Please ensure your scikit-learn version is up-to-date and "
                 "the model was saved with a recent version of scikit-learn.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
