import numpy as np
import pandas as pd
import tensorflow as tf 
import joblib 
import streamlit as st 

#load model and scaler 
model=tf.keras.models.load_model('Diabetic_model.h5')
scaler=joblib.load('scaler (1).pkl')

# app page
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("enter the following details to predict diabetes or not")

#input fields 
pregnancies = st.number_input("enter the no of pregnancies",min_value=0,max_value=10,value=1)
glucose = st.number_input("enter the glucose level",min_value=0)
blood_pressure = st.number_input("enter the blood pressure",min_value=0)
skin_thickness = st.number_input("enter the skin thickness",min_value=0)
insulin = st.number_input("enter the insulin level",min_value=0)
bmi = st.number_input("enter the body weight of patient",min_value=1)
diabetespedegree_function = st.number_input("enter the diabetes pedigree function",min_value=0)
age = st.number_input("enter the age of patient",min_value=0)

#make prediction
if st.button("predict diabetes"):
  input_data = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetespedegree_function,age]])
  input_scaled = scaler.transform(input_data)
  prediction = model.predict(input_scaled)
  result = "Not diabetic" if prediction[0][0]<0.5 else "Diabetic"
  st.success(f"The person is {result}")