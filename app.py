from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string
from pathlib import Path

# Paths
ROOT = Path(".")
PROC = ROOT / "data" / "processed"
MODEL_FILE = PROC / "relevancy_model.pkl"
VECT_FILE = PROC / "vectorizer.pkl"

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(txt):
    if txt is None or pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z0-9\s/-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECT_FILE)

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        text = request.form.get("tender_text")
        clean = clean_text(text)
        vec = vectorizer.transform([clean])

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]

        prediction = "RELEVANT" if pred == 1 else "NOT RELEVANT"
        probability = round(float(proba) * 100, 2)     # <-- FIXED

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
