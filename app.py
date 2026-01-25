import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

model = load_model("model.keras")
col_trans = None
with open("col_trans.pkl", "rb") as file:
    col_trans = pickle.load(file)

st.title("Churn prediction - ANN classifier")

credit_score = st.number_input("Credit score", 300, 900)
geography = st.selectbox(label="Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox(label="Gender", options=["Male", "Female"])
age = st.number_input("Age", 18, 92)
tenure = st.slider("Tenure", 0, 10, step=1)
balance = st.number_input("Balance", min_value=0, max_value=500000)
num_products = st.slider("Num of products", min_value=1, max_value=4, step=1)
has_cr_card = int(st.checkbox("Has credit card"))
is_active = int(st.checkbox("Is active member"))
estimated_salary = st.number_input("Estimated salary", 0, 300000)

test_data = {
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [estimated_salary]
}

transformed_data = col_trans.transform(pd.DataFrame(test_data))
predicted = model.predict(transformed_data)
pred_value = predicted[0][0]

st.text(f"Predicted value: {pred_value}")
if pred_value > 0.5:
    st.text("Customer is likely to churn")
else:
    st.text("Customer is not likely to churn")