import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stMetric {
    border-radius: 15px;
    padding: 15px;
    background-color: #262730;
}

.big-font {
    font-size: 40px !important;
    font-weight: bold;
    color: #4CAF50;
}

.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

.title {
    text-align: center;
    color: #4CAF50;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# =========================
# HEADER
# =========================
st.markdown('<div class="title">📊 Customer Churn Predictor</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Predict whether a customer is likely to leave the bank</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Customer Information")

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("Age", 18, 92, 35)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    value=100000.0
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

num_of_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.sidebar.selectbox(
    "Active Member",
    [0, 1]
)

# =========================
# CUSTOMER SUMMARY
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Age", age)
col2.metric("Credit Score", credit_score)
col3.metric("Balance", f"${balance:,.0f}")
col4.metric("Salary", f"${estimated_salary:,.0f}")

st.divider()

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🚀 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform(
        [[geography]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(
            ['Geography']
        )
    )

    input_data = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df],
        axis=1
    )

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    probability = prediction[0][0]

    st.divider()

    st.subheader("Prediction Result")

    st.progress(float(probability))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Retention Probability",
            f"{(1-probability)*100:.2f}%"
        )

    if probability > 0.5:
        st.error(
            f"⚠️ High Risk Customer ({probability*100:.2f}% chance of churn)"
        )
    else:
        st.success(
            f"✅ Customer Likely To Stay ({(1-probability)*100:.2f}% confidence)"
        )

    # Risk Category
    st.subheader("Risk Assessment")

    if probability < 0.3:
        st.success("🟢 LOW RISK")
    elif probability < 0.7:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")
