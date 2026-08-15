import streamlit as st
import requests

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Insurance Charges Predictor")
st.write("Predict medical insurance charges using a tuned XGBoost model served via FastAPI.")

# ---------------------------------------------------------
# API config (hardcoded — not user-editable)
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Enter customer details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=64, value=30)
    bmi = st.number_input("BMI", min_value=0.1, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

payload = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}

with st.expander("See request payload"):
    st.json(payload)

# ---------------------------------------------------------
# Call API and show prediction
# ---------------------------------------------------------
if st.button("Predict Charges", type="primary"):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prediction = result["predicted_charges"]
            st.success(f"### Estimated Insurance Charges: ${prediction:,.2f}")

            if smoker == "yes":
                st.info(
                    "Note: Smoking status has a strong impact on predicted charges "
                    "in this model — it's typically the single biggest cost driver."
                )
        elif response.status_code == 422:
            st.error("Validation error — check that your inputs match the expected format.")
            st.json(response.json())
        else:
            st.error(f"API returned status {response.status_code}")
            st.text(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Couldn't connect to the API at `{API_URL}`. "
            "Make sure your FastAPI server is running (`uvicorn api:app --reload`)."
        )
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API server may be slow to respond or unreachable.")