import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Weekday mapping (0=Sun, 1=Mon, ..., 6=Sat based on sklearn get_dummies)
weekday_dict = {
    'Sun': [1, 0, 0, 0, 0, 0, 0],
    'Mon': [0, 1, 0, 0, 0, 0, 0],
    'Tue': [0, 0, 1, 0, 0, 0, 0],
    'Wed': [0, 0, 0, 1, 0, 0, 0],
    'Thu': [0, 0, 0, 0, 1, 0, 0],
    'Fri': [0, 0, 0, 0, 0, 1, 0],
    'Sat': [0, 0, 0, 0, 0, 0, 1]
}

st.set_page_config(page_title="Bike Share Predictor", page_icon="🚴")

st.title("🚴 Bike Share Count Predictor")
st.markdown("Predict the number of bike rides based on time and day.")

# Inputs
hour = st.selectbox("🕐 Select Hour (0-23)", list(range(24)))
is_holiday = st.selectbox("🎉 Is it a Holiday?", ["No", "Yes"])
day_of_week = st.selectbox("📅 Select Day", ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])

if st.button("🔍 Predict Bike Count"):
    data = []

    # hour
    data.append(int(hour))

    # is_holiday (get_dummies: is_holiday_0 = not holiday, is_holiday_1 = holiday)
    if is_holiday == "Yes":
        data.extend([0, 1])  # is_holiday_0=0, is_holiday_1=1
    else:
        data.extend([1, 0])  # is_holiday_0=1, is_holiday_1=0

    # weekday_0 to weekday_6
    data.extend(weekday_dict[day_of_week])

    prediction = int(model.predict([data])[0])

    st.success(
        f"🚲 Predicted bike rides on **{day_of_week}** at **{hour}:00 Hrs** = **{prediction}**"
    )
