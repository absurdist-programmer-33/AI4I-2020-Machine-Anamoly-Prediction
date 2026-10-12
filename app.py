import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

st.set_page_config(page_title="Machine Anomaly Detection")
st.title("AI4I Machine Anomaly Detection")

#Loading data from ai4i2020 
@st.cache_data
def load_data():
    return pd.read_csv("ai4i2020.csv")

try:
    df = load_data()
except Exception:
    st.error("Dataset file ai4i2020.csv missing from the GitHub repository.")
    st.stop()

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

if not all(col in df.columns for col in features):
    st.error("CSV columns do not match the original AI4I dataset.")
    st.stop()

normal_data = df[features].copy()

if "Machine failure" in df.columns:
    normal_data = df.loc[df["Machine failure"] == 0, features].copy()

normal_data = normal_data.apply(
    pd.to_numeric, errors="coerce"
).dropna()

scaler = StandardScaler()
X_train = scaler.fit_transform(normal_data)

model = LocalOutlierFactor(
    n_neighbors=min(20, len(X_train) - 1),
    contamination=0.03,
    novelty=True
)
model.fit(X_train)

st.caption(f"Dataset loaded: {len(df)} records")

st.subheader("Enter Sensor Values")

air = st.number_input("Air temperature [K]", value=300.0)
process = st.number_input("Process temperature [K]", value=310.0)
rpm = st.number_input("Rotational speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
wear = st.number_input("Tool wear [min]", value=100)

if st.button("Predict"):
    values = [[air, process, rpm, torque, wear]]
    scaled_values = scaler.transform(values)
    result = model.predict(scaled_values)[0]

    if result == -1:
        st.error("ANOMALY")
    else:
        st.success("NORMAL")
