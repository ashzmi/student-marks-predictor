
import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Students Marks Predictor",
    page_icon="🐲",
    layout="centered"
)

st.title("STUDENTS MARKS PREDICTOR")
st.write("Enter The Number Of Hours Studied (1-10) And **Click Predict** To See The Predicted")

# Load The Model
def load_model(model):
    with open(model, "rb") as f:
        slr = pickle.load(f)
    return slr

try:
    model = load_model("slr.pkl")
except Exception as e:
    st.error("Your Pickle File Not Found")
    st.exception(e)
    st.stop()

hours = st.number_input(
    "Hours Studied",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    format="%.1f"
)

if st.button("Predict"):
    try:
        x = np.array([[hours]])
        predictions = model.predict(x)
        predictions = predictions[0]

        st.success(f"Predicted Marks : {predictions:.1f}")
        st.write("Note : This Is ML Model Prediction **Results May Vary**")

    except Exception as e:
        st.error(f"Prediction Failed : {e}")

