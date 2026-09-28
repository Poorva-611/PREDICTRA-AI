from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ===================================
# CREATE FASTAPI APPLICATION
# ===================================

app = FastAPI(
    title="PREDICTRA AI API",
    description="Predictive Maintenance Machine Learning API",
    version="1.0"
)


# ===================================
# LOAD TRAINED ML MODEL
# ===================================

model = joblib.load(
    "model/predictive_maintenance_model.pkl"
)


# ===================================
# INPUT DATA FORMAT
# ===================================

class SensorData(BaseModel):

    temperature: float
    vibration: float
    current: float
    rpm: float


# ===================================
# ROOT ENDPOINT
# ===================================

@app.get("/")
def home():

    return {
        "message": "PREDICTRA AI API is running"
    }


# ===================================
# MAINTENANCE ANALYSIS
# ===================================

def generate_maintenance_note(
    temperature,
    vibration,
    current,
    rpm,
    status
):

    issues = []


    # -----------------------------------
    # TEMPERATURE ANALYSIS
    # -----------------------------------

    if temperature >= 60:

        issues.append(
            "High temperature: inspect cooling, "
            "ventilation, and possible overheating."
        )

    elif temperature >= 50:

        issues.append(
            "Elevated temperature: monitor cooling "
            "and operating conditions."
        )


    # -----------------------------------
    # VIBRATION ANALYSIS
    # -----------------------------------

    if vibration >= 3.0:

        issues.append(
            "High vibration: inspect bearings, "
            "shaft alignment, mounting, and mechanical imbalance."
        )

    elif vibration >= 2.0:

        issues.append(
            "Elevated vibration: inspect machine mounting, "
            "alignment, and mechanical components."
        )


    # -----------------------------------
    # CURRENT ANALYSIS
    # -----------------------------------

    if current >= 4.5:

        issues.append(
            "High current: check motor overload, "
            "mechanical load, and electrical connections."
        )

    elif current >= 3.8:

        issues.append(
            "Elevated current: monitor motor load "
            "and electrical operating conditions."
        )


    # -----------------------------------
    # RPM ANALYSIS
    # -----------------------------------

    if rpm < 1200:

        issues.append(
            "Low RPM: inspect mechanical load, "
            "shaft condition, and motor performance."
        )

    elif rpm > 1800:

        issues.append(
            "High RPM: check speed control "
            "and motor operating conditions."
        )


    # ===================================
    # STATUS-BASED RECOMMENDATION
    # ===================================

    if status == "Healthy":

        if issues:

            maintenance_note = (
                "Machine is currently classified as HEALTHY, "
                "but some sensor parameters require monitoring. "
                "Continue normal operation while observing the "
                "identified conditions."
            )

        else:

            maintenance_note = (
                "No significant abnormal conditions detected. "
                "Continue normal operation and routine monitoring."
            )


    elif status == "Warning":

        if issues:

            maintenance_note = (
                "Moderate failure risk detected. "
                "Schedule preventive maintenance and inspect "
                "the identified machine conditions."
            )

        else:

            maintenance_note = (
                "Moderate failure risk detected. "
                "Schedule a preventive maintenance inspection."
            )


    else:

        if issues:

            maintenance_note = (
                "High failure risk detected. "
                "Perform a detailed maintenance inspection "
                "before continued operation."
            )

        else:

            maintenance_note = (
                "High failure risk detected. "
                "Perform a detailed maintenance inspection."
            )


    # ===================================
    # IF NO SENSOR ISSUE IS DETECTED
    # ===================================

    if not issues:

        issues.append(
            "No specific sensor abnormality detected."
        )


    return maintenance_note, issues


# ===================================
# PREDICTION ENDPOINT
# ===================================

@app.post("/predict")
def predict(data: SensorData):


    # -----------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------

    input_data = pd.DataFrame({

        "temperature": [data.temperature],

        "vibration": [data.vibration],

        "current": [data.current],

        "rpm": [data.rpm]

    })


    # -----------------------------------
    # ML PREDICTION
    # -----------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # -----------------------------------
    # FAILURE PROBABILITY
    # -----------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    failure_percentage = probability * 100


    # -----------------------------------
    # MACHINE STATUS
    # -----------------------------------

    if failure_percentage < 40:

        status = "Healthy"

    elif failure_percentage < 70:

        status = "Warning"

    else:

        status = "Critical"


    # -----------------------------------
    # MAINTENANCE ANALYSIS
    # -----------------------------------

    maintenance_note, recommendations = (
        generate_maintenance_note(
            data.temperature,
            data.vibration,
            data.current,
            data.rpm,
            status
        )
    )


    # -----------------------------------
    # API RESPONSE
    # -----------------------------------

    return {

        "prediction":
            "Failure" if prediction == 1 else "Normal",

        "failure_probability":
            round(failure_percentage, 2),

        "status":
            status,

        "maintenance_note":
            maintenance_note,

        "recommendations":
            recommendations
    }
