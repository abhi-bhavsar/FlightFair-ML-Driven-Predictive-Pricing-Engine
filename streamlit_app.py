import os
import pickle
from datetime import datetime, timedelta

import streamlit as st

from src.data_pipeline import preprocess_flight_data


st.set_page_config(
    page_title="Flight Fare Prediction",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

.main-title{
    font-size:2.5rem;
    font-weight:700;
    margin-bottom:0;
}

.subtitle{
    color:#888;
    font-size:1rem;
    margin-bottom:2rem;
}

.stButton>button{
    width:100%;
    height:3.2rem;
    font-size:18px;
    font-weight:600;
}

div[data-testid="stMetric"]{
    border-radius:12px;
    padding:18px;
}

</style>
""", unsafe_allow_html=True)
MODEL_PATH = "model/best_xgboost_flight_model.pkl"

AIRLINES = [
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy"
]

SOURCES = [
    "Delhi",
    "Kolkata",
    "Mumbai",
    "Chennai"
]

DESTINATIONS = [
    "Cochin",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "New Delhi"
]
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


try:

    model = load_model()

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")
    st.stop()
st.markdown(
    "<div class='main-title'>✈️ Flight Fare Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict airline ticket prices using an XGBoost Machine Learning model.</div>",
    unsafe_allow_html=True
)

st.divider()
st.subheader("Flight Information")

left, right = st.columns(2)

with left:

    airline = st.selectbox(
        "Airline",
        AIRLINES
    )

with right:

    stops = st.selectbox(
        "Total Stops",
        [0,1,2,3,4],
        index=0
    )


left, right = st.columns(2)

with left:

    source = st.selectbox(
        "Source",
        SOURCES
    )

with right:

    destination = st.selectbox(
        "Destination",
        DESTINATIONS
    )

st.divider()

st.subheader("Travel Schedule")

today = datetime.today()

default_departure = today + timedelta(days=7)

default_arrival = default_departure + timedelta(hours=2)


left, right = st.columns(2)

with left:

    dep_date = st.date_input(
        "Departure Date",
        value=default_departure.date(),
        min_value=today.date()
    )

    dep_time = st.time_input(
        "Departure Time",
        value=default_departure.time().replace(second=0,microsecond=0)
    )

with right:

    arr_date = st.date_input(
        "Arrival Date",
        value=default_arrival.date(),
        min_value=dep_date
    )

    arr_time = st.time_input(
        "Arrival Time",
        value=default_arrival.time().replace(second=0,microsecond=0)
    )

st.divider()
if st.button("🚀 Predict Flight Fare"):

    departure_datetime = datetime.combine(dep_date, dep_time)
    arrival_datetime = datetime.combine(arr_date, arr_time)

    if arrival_datetime <= departure_datetime:
        arrival_datetime += timedelta(days=1)

    duration = arrival_datetime - departure_datetime
    duration_hours = duration.total_seconds() / 3600

    if source == destination:

        st.error("❌ Source and Destination cannot be the same.")
        st.stop()

    if duration_hours < 0.30:

        st.error("❌ Flight duration is unrealistically short.")
        st.stop()

    if stops >= 3 and duration_hours < 4:

        st.error(
            "❌ Flights having 3 or more stops generally require longer travel durations."
        )
        st.stop()

    if duration_hours > 48:

        st.error(
            "❌ Flight duration appears invalid."
        )
        st.stop()

    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------

    with st.spinner("Predicting flight fare..."):

        try:

            input_df = preprocess_flight_data(
                airline=airline,
                source=source,
                destination=destination,
                departure_datetime=departure_datetime,
                arrival_datetime=arrival_datetime,
                stops=stops
            )

            prediction = float(model.predict(input_df)[0])

        except Exception as e:

            st.error(f"Prediction Failed\n\n{e}")
            st.stop()
    if prediction < 0:

        prediction = abs(prediction)

    prediction = max(prediction, 1200)

    prediction = round(prediction, 2)
    st.success("Prediction Generated Successfully!")

    st.metric(
        label="💰 Estimated Flight Fare",
        value=f"₹ {prediction:,.2f}"
    )

    st.divider()

    st.subheader("Journey Summary")

    c1, c2 = st.columns(2)

    with c1:

        st.write(f"**✈️ Airline** : {airline}")
        st.write(f"**🛫 Source** : {source}")
        st.write(f"**🛬 Destination** : {destination}")
        st.write(f"**🛑 Stops** : {stops}")

    with c2:

        st.write(
            f"**📅 Departure** : {departure_datetime.strftime('%d %b %Y, %I:%M %p')}"
        )

        st.write(
            f"**📅 Arrival** : {arrival_datetime.strftime('%d %b %Y, %I:%M %p')}"
        )

        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)

        st.write(
            f"**⏱ Duration** : {hours}h {minutes}m"
        )

    st.divider()

    st.info(
        "💡 This fare is an estimate generated by the trained XGBoost regression model. Actual airline prices may vary depending on seat availability, demand, holidays, airline policies, and booking time."
    )