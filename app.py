from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model and encoders
with open('hospital_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)
    le_bpjs = pickle.load(file)
    le_domisili = pickle.load(file)
    le_diagnosis = pickle.load(file)
    le_specialist = pickle.load(file)
    le_rumah_sakit = pickle.load(file)

# Mapping keywords in diagnosis to specialists
diagnosis_to_specialist = {
    "Saraf": "Neurologi (Saraf)",
    "Otot": "Miologi (Otot)",
    "Sendi": "Arthrologi (Sendi)",
    "Janin": "Fetologi (Janin)",
    "Embrio": "Embriologi (Embrio)",
    "Darah": "Hematologi (Darah)",
    "Tulang": "Osteologi (Tulang)",
    "Jantung": "Kardiologi (Jantung)",
    "Ginjal": "Nefrologi (Ginjal)",
    "Sel": "Sitologi (Sel)",
    "Parasit": "Parasitologi (Parasit)",
    "Imun": "Imunologi (Imun)",
    "Kanker": "Onkologi (Kanker)",
    "Reproduksi": "Ginekologi (Reproduksi)"
}

# Function to map diagnosis input to specialist
def map_diagnosis_to_specialist(diagnosis):
    for keyword, specialist in diagnosis_to_specialist.items():
        if keyword.lower() in diagnosis.lower():
            return specialist
    return "Tidak Diketahui"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommended_hospital = None
    specialist_for_hospital = None
    error = None

    if request.method == "POST":
        # Get user input
        bpjs_input = request.form['bpjs']
        umur_input = int(request.form['umur'])
        domisili_input = request.form['domisili']
        diagnosis_input = request.form['diagnosis']

        # Map diagnosis to specialist
        specialist_input = map_diagnosis_to_specialist(diagnosis_input)

        if specialist_input == "Tidak Diketahui":
            error = "Penyakit Anda termasuk dalam jenis yang tidak perlu dirujuk"
        else:
            # Encode inputs
            bpjs_encoded = le_bpjs.transform([bpjs_input])[0]
            domisili_encoded = le_domisili.transform([domisili_input])[0]
            specialist_encoded = le_specialist.transform([specialist_input])[0]

            # Predict recommended hospital
            prediction = model.predict(
                np.array([[bpjs_encoded, umur_input, domisili_encoded, 0, specialist_encoded]])
            )
            recommended_hospital = le_rumah_sakit.inverse_transform(prediction)[0]
            specialist_for_hospital = specialist_input

    return render_template('index.html', recommended_hospital=recommended_hospital, 
                           specialist_for_hospital=specialist_for_hospital, error=error)

if __name__ == "__main__":
    app.run(debug=True)
