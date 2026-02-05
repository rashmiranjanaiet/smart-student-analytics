from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        internal_marks = float(request.form["internal_marks"])
        assignment_score = float(request.form["assignment_score"])

        features = np.array([[study_hours, attendance, internal_marks, assignment_score]])
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

