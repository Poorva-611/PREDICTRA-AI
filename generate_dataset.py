import numpy as np
import pandas as pd

np.random.seed(42)

# Number of samples
n_normal = 8000
n_failure = 2000

# -----------------------------
# NORMAL MACHINE DATA
# -----------------------------
normal_temperature = np.random.normal(45, 4, n_normal)
normal_vibration = np.random.normal(1.4, 0.25, n_normal)
normal_current = np.random.normal(3.0, 0.35, n_normal)
normal_rpm = np.random.normal(1500, 80, n_normal)

normal_failure = np.zeros(n_normal, dtype=int)


# -----------------------------
# FAILURE MACHINE DATA
# -----------------------------
failure_temperature = np.random.normal(65, 7, n_failure)
failure_vibration = np.random.normal(3.0, 0.6, n_failure)
failure_current = np.random.normal(4.5, 0.7, n_failure)
failure_rpm = np.random.normal(1250, 180, n_failure)

failure_label = np.ones(n_failure, dtype=int)


# -----------------------------
# COMBINE DATA
# -----------------------------
data = pd.DataFrame({
    "temperature": np.concatenate(
        [normal_temperature, failure_temperature]
    ),

    "vibration": np.concatenate(
        [normal_vibration, failure_vibration]
    ),

    "current": np.concatenate(
        [normal_current, failure_current]
    ),

    "rpm": np.concatenate(
        [normal_rpm, failure_rpm]
    ),

    "machine_failure": np.concatenate(
        [normal_failure, failure_label]
    )
})


# Shuffle the dataset
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Round sensor values
data["temperature"] = data["temperature"].round(2)
data["vibration"] = data["vibration"].round(2)
data["current"] = data["current"].round(2)
data["rpm"] = data["rpm"].round().astype(int)


# Save dataset
data.to_csv(
    "data/machine_sensor_data.csv",
    index=False
)


print("\n===================================")
print("   PREDICTRA AI DATASET GENERATED")
print("===================================")

print(f"Total samples: {len(data)}")

print("\nFailure distribution:")
print(data["machine_failure"].value_counts())

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset saved successfully:")
print("data/machine_sensor_data.csv")

print("===================================")