import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="crdeepa/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction App")
st.write("""
The Tourism Package Prediction App for Visit With Us predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

Age=st.number_input("Age", min_value=18, max_value=100, value=41, step=1)
CityTier=st.number_input("CityTier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch=st.number_input("DurationOfPitch", min_value=5, max_value=127, value=15, step=1)
NumberOfPersonVisiting=st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=5, step=1)
NumberOfFollowups=st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=5, step=1)
PreferredPropertyStar=st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3, step=1)
NumberOfTrips=st.number_input("NumberOfTrips", min_value=1, max_value=30, value=5, step=1)
Passport=st.number_input("Passport", min_value=0, max_value=1, value=0, step=1)
PitchSatisfactionScore=st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3, step=1)
OwnCar=st.number_input("OwnCar", min_value=0, max_value=1, value=0, step=1)
NumberOfChildrenVisiting=st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=1, step=1)
MonthlyIncome=st.number_input("MonthlyIncome", min_value=1000, max_value=100000, value=50000, step=1)

TypeofContact=st.selectbox("TypeofContact", ["Company Invited Other","Company Invited"])
Occupation=st.selectbox("Occupation", ["Salaried","Free Lancer","Small Business","Large Business"])
Gender=st.selectbox("Gender", ["Male","Female"])
ProductPitched=st.selectbox("ProductPitched", ["Deluxe","Basic","Standard","Super Deluxe","King"])
MaritalStatus=st.selectbox("MaritalStatus", ["Married","Single","Divorced","Unmarried"])
Designation=st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,
    'TypeofContact':TypeofContact,
    'Occupation':Occupation,
    'Gender':Gender,
    'ProductPitched':ProductPitched,
    'MaritalStatus':MaritalStatus,
    'Designation':Designation
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Prediction: {prediction}")    
