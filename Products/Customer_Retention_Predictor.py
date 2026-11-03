# -*- coding: utf-8 -*-
"""
Customer Churn Predictor - Streamlit app.

Serves the tuned LightGBM pipeline (ROS + SelectKBest) saved from the training
notebook as `final_tuned_lightgbm_ros_selectkbest.pkl`.
"""

import os
import pickle
import warnings

import pandas as pd
import streamlit as st

# The fitted pipeline hands plain ndarrays to the classifier after SelectKBest,
# so sklearn emits a harmless "X does not have valid feature names" warning.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

MODEL_FILENAME = "final_tuned_lightgbm_ros_selectkbest.pkl"

# The training notebook labelled the NumberOfAddress bins with an EN DASH
# (U+2013), not a hyphen. The BinaryEncoder matches these strings exactly, so an
# ASCII "-" here would silently fall through to the unknown category and distort
# every prediction. Build them from an escape so no editor or git text
# normalisation can mangle the character.
EN_DASH = "\u2013"
ADDRESS_BINS = [f"1{EN_DASH}2", "3", f"4{EN_DASH}6", "7+"]

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

# Main title of the web app
st.title("Customer Churn Predictor")
st.text(
    "This application predicts the likelihood of a customer churning "
    "based on their behavior and preferences."
)

# Sidebar for user input
st.sidebar.header("Please input customer features")


def create_user_input():
    """
    Create interactive widgets in the sidebar to collect input features from the user.
    Returns a single-row DataFrame containing the customer's data.
    """

    # Numeric inputs
    Tenure = st.sidebar.slider('Tenure (in months)', 0, 61, 12)
    WarehouseToHome = st.sidebar.slider('Warehouse To Home (in km)', 5, 126, 20)
    HourSpendOnApp = st.sidebar.slider('Hours Spent on App (per day)', 0.0, 5.0, 1.0)
    NumberOfDeviceRegistered = st.sidebar.slider('Number of Devices Registered', 1, 6, 2)
    OrderAmountHikeFromlastYear = st.sidebar.slider('Order Amount Hike From Last Year (%)', 11, 26, 15)
    CouponUsed = st.sidebar.slider('Coupons Used', 0, 16, 3)
    OrderCount = st.sidebar.slider('Order Count', 1, 16, 5)
    DaySinceLastOrder = st.sidebar.slider('Days Since Last Order', 0, 46, 10)
    CashbackAmount = st.sidebar.slider('Cashback Amount', 0.0, 324.99, 50.0)

    # Categorical features
    PreferredLoginDevice = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Computer'])
    CityTier = st.sidebar.selectbox('City Tier', ['1', '2', '3'])
    PreferredPaymentMode = st.sidebar.selectbox(
        'Preferred Payment Mode',
        ['Debit Card', 'Credit Card', 'E wallet', 'Cash on Delivery', 'UPI']
    )
    Gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    PreferedOrderCat = st.sidebar.selectbox(
        'Preferred Order Category',
        ['Mobile Phone', 'Laptop & Accessory', 'Grocery', 'Fashion', 'Others']
    )
    SatisfactionScore = st.sidebar.selectbox('Satisfaction Score', ['1', '2', '3', '4', '5'])
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    # Kept so the column layout matches how the pipeline was fitted. The fitted
    # ColumnTransformer uses remainder='drop', and Complain is the dropped
    # column, so this widget does not influence the prediction.
    Complain = st.sidebar.radio('Customer Complaint?', [0, 1])
    CountOfAddress = st.sidebar.selectbox('Count of Address', ADDRESS_BINS)

    # Collect inputs
    user_data = {
        'Tenure': Tenure,
        'WarehouseToHome': WarehouseToHome,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'CouponUsed': CouponUsed,
        'OrderCount': OrderCount,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier': CityTier,
        'PreferredPaymentMode': PreferredPaymentMode,
        'Gender': Gender,
        'PreferedOrderCat': PreferedOrderCat,
        'SatisfactionScore': SatisfactionScore,
        'MaritalStatus': MaritalStatus,
        'Complain': Complain,
        'CountOfAddress': CountOfAddress
    }

    return pd.DataFrame([user_data])


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path):
    """Unpickle the trained pipeline once per session instead of on every rerun."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


# Resolve the model next to this script so the app works from any CWD.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

try:
    model_loaded = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("Error: Model file not found.")
    st.write(f"Looking for file at: {MODEL_PATH}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Get input data
data_customer = create_user_input()

# The fitted ColumnTransformer selects its passthrough/drop columns by POSITION,
# so the frame must carry exactly the training columns in the training order.
expected_features = list(getattr(model_loaded, 'feature_names_in_', data_customer.columns))
missing = [c for c in expected_features if c not in data_customer.columns]
if missing:
    st.error(f"Input is missing features the model expects: {missing}")
    st.stop()
data_customer = data_customer[expected_features]

# Display input.
# Note: transposing a single-row frame collapses ints, floats and strings into one
# object column, which Arrow cannot serialise ("Could not convert 'Mobile Phone'
# with type str: tried to convert to double"). Streamlit currently papers over
# that with an internal cast-to-string fallback, but it is not a contract worth
# depending on, so cast here and label the columns explicitly.
st.subheader("Customer's Features")
features_view = data_customer.transpose().reset_index()
features_view.columns = ['Feature', 'Value']
features_view['Value'] = features_view['Value'].astype(str)
st.dataframe(features_view, hide_index=True, use_container_width=True)

# Predict churn
try:
    kelas = model_loaded.predict(data_customer)
    prob = model_loaded.predict_proba(data_customer)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Show result
st.subheader("Prediction Result")

if int(kelas[0]) == 1:
    st.error("Churn: **Yes** – This customer is likely to churn.")
else:
    st.success("Churn: **No** – This customer is likely to stay.")

churn_prob = float(prob[1])
st.metric("Probability of Churn", f"{churn_prob:.2%}")
st.progress(min(max(churn_prob, 0.0), 1.0))
