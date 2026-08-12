"""
train_mlflow.py
Same Animal Classifier as train.py, but now logs every run to MLflow
so we can track and compare experiments over time.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 1. Our animal dataset (same as before)
data = [
    ("Lion",       1, 0, 0, 0, 1, 0, 4, "Mammal"),
    ("Dog",        1, 0, 0, 0, 1, 0, 4, "Mammal"),
    ("Elephant",   1, 0, 0, 0, 1, 0, 4, "Mammal"),
    ("Bat",        1, 1, 0, 0, 1, 0, 2, "Mammal"),
    ("Whale",      0, 0, 0, 1, 1, 0, 0, "Mammal"),
    ("Human",      1, 0, 0, 0, 1, 0, 2, "Mammal"),
    ("Rabbit",     1, 0, 0, 0, 1, 0, 4, "Mammal"),
    ("Eagle",      0, 1, 1, 0, 1, 0, 2, "Bird"),
    ("Sparrow",    0, 1, 1, 0, 1, 0, 2, "Bird"),
    ("Penguin",    0, 0, 1, 0, 1, 0, 2, "Bird"),
    ("Ostrich",    0, 0, 1, 0, 1, 0, 2, "Bird"),
    ("Owl",        0, 1, 1, 0, 1, 0, 2, "Bird"),
    ("Salmon",     0, 0, 1, 1, 0, 0, 0, "Fish"),
    ("Shark",      0, 0, 1, 1, 0, 0, 0, "Fish"),
    ("Goldfish",   0, 0, 1, 1, 0, 0, 0, "Fish"),
    ("Tuna",       0, 0, 1, 1, 0, 0, 0, "Fish"),
    ("Snake",      0, 0, 1, 0, 1, 0, 0, "Reptile"),
    ("Lizard",     0, 0, 1, 0, 1, 0, 4, "Reptile"),
    ("Crocodile",  0, 0, 1, 0, 1, 0, 4, "Reptile"),
    ("Turtle",     0, 0, 1, 0, 1, 0, 4, "Reptile"),
    ("Frog",       0, 0, 1, 0, 1, 1, 4, "Amphibian"),
    ("Salamander", 0, 0, 1, 0, 1, 1, 4, "Amphibian"),
    ("Toad",       0, 0, 1, 0, 1, 1, 4, "Amphibian"),
    ("Newt",       0, 0, 1, 0, 1, 1, 4, "Amphibian"),
    ("Bee",        0, 1, 1, 0, 1, 0, 6, "Insect"),
    ("Ant",        0, 0, 1, 0, 1, 0, 6, "Insect"),
    ("Butterfly",  0, 1, 1, 0, 1, 0, 6, "Insect"),
    ("Beetle",     0, 1, 1, 0, 1, 0, 6, "Insect"),
]

columns = ["name", "has_fur", "can_fly", "lays_eggs", "has_fins",
           "breathes_air", "moist_skin", "legs", "class"]
df = pd.DataFrame(data, columns=columns)

X = df.drop(columns=["name", "class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Set these two values to experiment with different settings
N_ESTIMATORS = 50   # try changing this: 50, 100, 200...
MAX_DEPTH = None      # try changing this: 3, 5, 10...

# 3. Start an MLflow run - everything inside this 'with' block gets logged
with mlflow.start_run():

    # Log the settings (parameters) we used for this run
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate it
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")

    # Log the result (metric) of this run
    mlflow.log_metric("accuracy", accuracy)

    # Log the actual trained model as part of this run
    mlflow.sklearn.log_model(model, "model")

    print("Run logged to MLflow!")