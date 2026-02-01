import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color:#0f172a;
}
.main {
    background-color:#0f172a;
}
h1 {
    text-align:center;
    color:#38bdf8;
    font-size:2.5rem;
}
.subtitle {
    text-align:center;
    color:#94a3b8;
    font-size:1.1rem;
    margin-bottom:2rem;
}
.block-container {
    padding:2rem 4rem;
    max-width:1400px;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color:#e2e8f0 !important;
    font-weight:600 !important;
    font-size:1rem !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background-color:#1e293b !important;
    border:2px solid #334155 !important;
    border-radius:8px !important;
    color:#e2e8f0 !important;
    font-size:1rem !important;
    height:3rem !important;
}
.stButton>button {
    background-color:#38bdf8;
    color:black;
    font-size:20px;
    font-weight:600;
    border-radius:8px;
    height:3.5em;
    width:100%;
    margin-top:2rem;
}
.stButton>button:hover {
    background-color:#0ea5e9;
}
.stButton>button:disabled {
    background-color:#64748b;
    cursor:not-allowed;
}
.result-box {
    padding:2rem;
    border-radius:10px;
    font-size:22px;
    text-align:center;
    margin-top:2rem;
    font-weight:600;
}
.churn {
    background-color:#fee2e2;
    color:#b91c1c;
}
.no-churn {
    background-color:#dcfce7;
    color:#166534;
}
.warning {
    background-color:#fef3c7;
    color:#92400e;
}
hr {
    margin:2rem 0;
    border-color:#334155;
}
</style>
""",unsafe_allow_html=True)

model=tf.keras.models.load_model('model.h5')
with open('label_encoder.pkl','rb')as file:
    label_encoder=pickle.load(file)
with open('ohe.pkl','rb')as file:
    ohe=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

st.title("📉 Customer Churn Prediction App")
st.markdown('<p class="subtitle">Predict whether a bank customer is likely to leave or stay, using a trained ANN model.</p>',unsafe_allow_html=True)
st.divider()

col1,col2=st.columns(2)

with col1:
    geography=st.selectbox("🌍 Geography",ohe.categories_[0])
    gender=st.selectbox("👤 Gender",label_encoder.classes_)
    age=st.slider("🎂 Age (years)",18,92,35)
    credit_score=st.number_input("💳 Credit Score",min_value=300,max_value=900,value=None,placeholder="Enter credit score")
    balance=st.number_input("💰 Account Balance (€)",min_value=0.0,step=1000.0,value=None,placeholder="Enter balance")

with col2:
    estimated_salary=st.number_input("🏦 Estimated Salary (€)",min_value=0.0,step=1000.0,value=None,placeholder="Enter salary")
    tenure=st.slider("📆 Tenure (years with bank)",0,10,5)
    num_of_products=st.slider("📦 Number of Bank Products",1,4,2)
    has_cr_card=st.selectbox("💳 Has Credit Card?",[0,1],format_func=lambda x: "Yes" if x==1 else "No")
    is_active_member=st.selectbox("⚡ Is Active Member?",[0,1],format_func=lambda x: "Yes" if x==1 else "No")

st.divider()

is_valid=credit_score is not None and balance is not None and estimated_salary is not None

if st.button("🔍 Predict Churn",disabled=not is_valid):
    if not is_valid:
        st.markdown(
            '<div class="result-box warning">⚠️ Please fill in all required fields</div>',
            unsafe_allow_html=True
        )
    else:
        input_df=pd.DataFrame({
            'CreditScore':[credit_score],
            'Gender':[label_encoder.transform([gender])[0]],
            'Age':[age],
            'Tenure':[tenure],
            'Balance':[balance],
            'NumOfProducts':[num_of_products],
            'HasCrCard':[has_cr_card],
            'IsActiveMember':[is_active_member],
            'EstimatedSalary':[estimated_salary]
        })

        geo_encoded=ohe.transform(pd.DataFrame({'Geography':[geography]})).toarray()
        geo_encoded_df=pd.DataFrame(
            geo_encoded,
            columns=ohe.get_feature_names_out(['Geography'])
        )

        input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

        scaled_data=scaler.transform(input_df)
        prediction=model.predict(scaled_data)
        prediction_prob=prediction[0][0]

        st.markdown(f"### 📊 Churn Probability: **{prediction_prob:.2%}**")

        if prediction_prob>0.5:
            st.markdown(
                '<div class="result-box churn">❌ Customer is likely to CHURN</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box no-churn">✅ Customer is likely to STAY</div>',
                unsafe_allow_html=True
            )

if not is_valid:
    st.info("ℹ️ Please enter Credit Score, Account Balance, and Estimated Salary to enable prediction")