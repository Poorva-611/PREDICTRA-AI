import streamlit as st
import pandas as pd
import requests


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="PREDICTRA AI",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------------
# FASTAPI BACKEND URL
# -----------------------------------
API_URL = "http://127.0.0.1:8000/predict"


# -----------------------------------
# HEADER
# -----------------------------------
st.title("⚙️ PREDICTRA AI")
st.subheader("AI-Powered Predictive Maintenance System")

st.write(
    "Analyze machine sensor parameters, predict failure risk, "
    "and receive maintenance recommendations."
)

st.divider()


# -----------------------------------
# SENSOR INPUTS
# -----------------------------------
st.header("🔧 Machine Sensor Parameters")

col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=20.0,
        max_value=100.0,
        value=45.0,
        step=0.5
    )

    vibration = st.number_input(
        "📳 Vibration",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )


with col2:

    current = st.number_input(
        "⚡ Current (A)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )

    rpm = st.number_input(
        "🔄 RPM",
        min_value=500,
        max_value=2500,
        value=1500,
        step=10
    )


# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
st.divider()


if st.button(
    "🔍 Analyze Machine",
    use_container_width=True
):

    # -----------------------------------
    # PREPARE SENSOR DATA
    # -----------------------------------

    sensor_data = {

        "temperature": temperature,

        "vibration": vibration,

        "current": current,

        "rpm": rpm
    }


    # -----------------------------------
    # SEND DATA TO FASTAPI
    # -----------------------------------

    try:

        response = requests.post(

            API_URL,

            json=sensor_data

        )


        # -----------------------------------
        # CHECK RESPONSE
        # -----------------------------------

        if response.status_code == 200:

            result = response.json()


            # -----------------------------------
            # GET RESULTS
            # -----------------------------------

            prediction = result["prediction"]

            failure_percentage = result["failure_probability"]

            status = result["status"]

            maintenance_note = result["maintenance_note"]

            recommendations = result["recommendations"]


            # -----------------------------------
            # PREDICTION RESULT
            # -----------------------------------

            st.header("📊 Prediction Result")


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(

                    "Failure Risk",

                    f"{failure_percentage:.2f}%"

                )


            with result_col2:

                st.metric(

                    "AI Prediction",

                    prediction.upper()

                )


            # -----------------------------------
            # MACHINE STATUS
            # -----------------------------------

            if status == "Healthy":

                st.success(
                    "🟢 Machine Status: HEALTHY"
                )

            elif status == "Warning":

                st.warning(
                    "🟡 Machine Status: WARNING"
                )

            else:

                st.error(
                    "🔴 Machine Status: CRITICAL"
                )


            # -----------------------------------
            # MAINTENANCE NOTE
            # -----------------------------------

            st.subheader("🛠️ Maintenance Recommendation")

            if status == "Healthy":

                st.info(
                    maintenance_note
                )

            elif status == "Warning":

                st.warning(
                    maintenance_note
                )

            else:

                st.error(
                    maintenance_note
                )


            # -----------------------------------
            # SPECIFIC RECOMMENDATIONS
            # -----------------------------------

            st.subheader("🔍 Areas to Inspect")


            for recommendation in recommendations:

                st.write(
                    "• " + recommendation
                )


            # -----------------------------------
            # SENSOR SUMMARY
            # -----------------------------------

            st.subheader("📋 Sensor Summary")


            summary = pd.DataFrame({

                "Parameter": [

                    "Temperature",

                    "Vibration",

                    "Current",

                    "RPM"

                ],

                "Value": [

                    f"{temperature:.2f} °C",

                    f"{vibration:.2f}",

                    f"{current:.2f} A",

                    f"{rpm} RPM"

                ]

            })


            st.table(summary)


        else:

            st.error(

                f"FastAPI returned an error: "
                f"{response.status_code}"

            )


    except requests.exceptions.ConnectionError:

        st.error(

            "❌ Cannot connect to FastAPI. "
            "Make sure the FastAPI server is running."

        )


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "PREDICTRA AI | AI-Based Predictive Maintenance "
    "System | Synthetic Sensor Dataset"
)