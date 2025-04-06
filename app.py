import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

# load the trained model
model = load_model('model.h5')

# Load Encoders and Scalars
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder_geo.pkl','rb') as file:
    ohe_encoder_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", ohe_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

# Input Dict
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'ExtimatedSalary': estimated_salary,
}

# Input DF
input_df = pd.DataFrame([input_data])

# Encoded DF
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
input_df[ohe_encoder_geo.get_feature_names_out()] = ohe_encoder_geo.transform(input_df[['Geography']])
input_df = input_df.drop('Geography', axis=1)
input_df = scaler.transform(input_df.values)

# Predict
prediction = model.predict(input_df)[0][0]
st.write(f"Churn Probability:{round(prediction*100,1)}%")
if prediction >= 0.5:
    st.write("Customer will Churn")
else:
    st.write("Customer will not Churn")



