"""
train.py
Animal Classifier - predicts an animal's class (Mammal, Bird, Fish, Reptile, Amphibian, Insect)
based on simple physical traits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Our animal dataset - traits (1 = yes, 0 = no) and legs count
# columns: name, has_fur, can_fly, lays_eggs, has_fins, breathes_air, moist_skin, legs, class
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

# 2. Split features (X) and label (y)
X = df.drop(columns=["name", "class"])
y = df["class"]

# 3. Split into train/test sets (stratify keeps class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate it
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# 6. Try it on a brand new "mystery animal" - traits of a bat
#    (fur, can fly, no eggs, breathes air, dry skin, 2 legs)
sample = pd.DataFrame([{
    "has_fur": 1, "can_fly": 1, "lays_eggs": 0,
    "has_fins": 0, "breathes_air": 1, "moist_skin": 0, "legs": 2
}])
print(f"Prediction for our mystery animal: {model.predict(sample)[0]}")

# 7. Save the trained model to disk
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")