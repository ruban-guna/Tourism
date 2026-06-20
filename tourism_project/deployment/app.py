import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download and load the model
model_path = hf_hub_download(repo_id="ruban-aiml/tourism_prediction_model", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase a tourism package based on their profile and interaction data.
Please enter the customer details below to get a prediction.
""")

# User input fields
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Other", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=2) # Default 3 star

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=5)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "Director", "Engineer", "Analyst", "Other"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=1000.0)

st.header("Customer Interaction Data")
col3, col4 = st.columns(2)

with col3:
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King", "Premium"])

with col4:
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=0.5)


# Assemble input into DataFrame, ensuring column order matches Xtrain
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0] # Probability of purchasing

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: Customer will purchase a package")
    else:
        st.info(f"The model predicts: Customer will NOT purchase a package")
