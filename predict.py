"""
predict.py
Loads the saved model.pkl and tests it on brand new animals
it has never seen before.
"""

import pandas as pd
import joblib

# 1. Load the trained model from disk
model = joblib.load("model.pkl")
print("Model loaded successfully!\n")

# 2. Define a few new "mystery animals" to test
# columns: has_fur, can_fly, lays_eggs, has_fins, breathes_air, moist_skin, legs
mystery_animals = {
    "Flying squirrel-like (fur, can glide, no eggs)": [1, 1, 0, 0, 1, 0, 4],
    "Ostrich-like (no fur, can't fly, lays eggs, 2 legs)": [0, 0, 1, 0, 1, 0, 2],
    "Eel-like (no fur, can't fly, lays eggs, has fins, lives in water)": [0, 0, 1, 1, 0, 0, 0],
    "Spider-like (no fur, can't fly, lays eggs, dry skin, 8 legs)": [0, 0, 1, 0, 1, 0, 8],
}

columns = ["has_fur", "can_fly", "lays_eggs", "has_fins", "breathes_air", "moist_skin", "legs"]

# 3. Run predictions
for description, traits in mystery_animals.items():
    sample = pd.DataFrame([traits], columns=columns)
    prediction = model.predict(sample)[0]
    print(f"{description}\n  --> Predicted class: {prediction}\n")