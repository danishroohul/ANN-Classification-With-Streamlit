import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the mode;
model = load_model('regression_model.h5')

# Load the scalers and encoders
with open("scaler_regression.pkl",'rb') as file:
    scaler = pickle.load(file)
with open("oh_encoder_regression.pkl",'rb') as file:
    oh_encoder = pickle.load(file)
with open("label_encoder_regression.pkl",'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit App
st.title("Customer Salary Predictor")

# User Input
geography = st.selectbox("Geography", oh_encoder.categories_[0])
gender = st.selectbox("Gender",label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox("Has Exited",[0,1])
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
    'Exited': exited,
}

# Input DF
input_df = pd.DataFrame([input_data])

# Encoded DF
input_df['Gender'] = label_encoder.transform(input_df['Gender'])
input_df[oh_encoder.get_feature_names_out()] = oh_encoder.transform(input_df[['Geography']])
input_df = input_df.drop('Geography', axis=1)
input_df = scaler.transform(input_df.values)

# Predict
prediction = model.predict(input_df)[0][0]
st.write(f"Customer's Salary:{round(prediction,1)}")
