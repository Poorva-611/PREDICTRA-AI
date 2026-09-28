import pandas as pd

import joblib

 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (

    RandomForestClassifier,

    GradientBoostingClassifier

)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix

)

 

 

# Load the dataset

data = pd.read_csv("data/machine_sensor_data.csv")

 

# Select input features and target

X = data[["temperature", "vibration", "current", "rpm"]]

y = data["machine_failure"]

 

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

 

 

# Create the models

models = {

    "Random Forest": RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced"

    ),

 

    "Logistic Regression": LogisticRegression(

        max_iter=1000,

        class_weight="balanced",

        random_state=42

    ),

 

    "Gradient Boosting": GradientBoostingClassifier(

        n_estimators=150,

        learning_rate=0.08,

        max_depth=3,

        random_state=42

    )

}

 

 

print("\n===================================")

print("       PREDICTRA AI MODEL")

print("===================================")

print("\nTraining and comparing models...\n")

 

 

best_model = None

best_model_name = ""

best_f1 = -1

 

 

# Train and evaluate every model

for name, model in models.items():

 

    model.fit(X_train, y_train)

 

    y_pred = model.predict(X_test)

 

    accuracy = accuracy_score(y_test, y_pred)

 

    report = classification_report(

        y_test,

        y_pred,

        target_names=["Normal", "Failure"],

        output_dict=True

    )

 

    failure_precision = report["Failure"]["precision"]

    failure_recall = report["Failure"]["recall"]

    failure_f1 = report["Failure"]["f1-score"]

 

    print("-----------------------------------")

    print(name)

    print("-----------------------------------")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(f"Failure Precision: {failure_precision * 100:.2f}%")

    print(f"Failure Recall: {failure_recall * 100:.2f}%")

    print(f"Failure F1-Score: {failure_f1 * 100:.2f}%")

 

    # Select the model with the best failure F1-score

    if failure_f1 > best_f1:

        best_f1 = failure_f1

        best_model = model

        best_model_name = name

 

 

# Final evaluation of the selected model

best_predictions = best_model.predict(X_test)

 

best_accuracy = accuracy_score(

    y_test,

    best_predictions

)

 

print("\n===================================")

print("          BEST MODEL")

print("===================================")

print(f"Selected Model: {best_model_name}")

print(f"Accuracy: {best_accuracy * 100:.2f}%")

 

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        best_predictions,

        target_names=["Normal", "Failure"]

    )

)

 

print("Confusion Matrix:")

print(confusion_matrix(y_test, best_predictions))

 

 

# Save the best model

model_path = "model/predictive_maintenance_model.pkl"

 

joblib.dump(best_model, model_path)

 

print("\n===================================")

print(f"Best model saved successfully:")

print(model_path)

print("===================================")

