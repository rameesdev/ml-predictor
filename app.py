from flask import Flask, request, jsonify, render_template
import joblib
import os
import math

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Define model directory
MODEL_DIR = "MODELS"

# Preload all models and encoders
models = {}
encoders = {}

# Income weightage mapping
INCOME_WEIGHTAGE = [
    (25000, 40), (50000, 39), (75000, 38), (100000, 37), (125000, 36), (150000, 35),
    (175000, 34), (200000, 33), (225000, 32), (250000, 31), (275000, 30), (300000, 29),
    (325000, 28), (350000, 27), (375000, 26), (400000, 25), (425000, 24), (450000, 23),
    (475000, 22), (500000, 21), (600000, 20), (700000, 19), (800000, 18), (900000, 17),
    (1000000, 16), (1100000, 14), (1200000, 12), (1300000, 10), (1400000, 8),
    (1500000, 6), (1600000, 4), (float('inf'), 3)
]

def get_income_weightage(income):
    for threshold, weightage in INCOME_WEIGHTAGE:
        if income <= threshold:
            return weightage
    return 3

# Distance weightage mapping
DISTANCE_WEIGHTAGE = {
    "Trivandrum": 20, "Kollam": 17.5, "Pathanamthitta": 16.6667, "Alappuzha": 13.75,
    "Kottayam": 14.5833, "Idukki": 14.79167, "Ernakulam": 11.6667, "Trichur": 9.16667,
    "Palakkad": 8.54167, "Malappuram": 7.5, "Calicut": 4.16667, "Kannur": 3.33333,
    "Kasaragod": 7.08333, "Wayanad": 1
}

def get_distance_weightage(district):
    return DISTANCE_WEIGHTAGE.get(district, 1)

# Load models
def load_models():
    for gender in ["female", "male"]:
        models[gender] = {}
        encoders[gender] = {}
        gender_path = os.path.join(MODEL_DIR, gender)
        if os.path.exists(gender_path):
            for semester in os.listdir(gender_path):
                semester_path = os.path.join(gender_path, semester)
                if os.path.isdir(semester_path):
                    models[gender][semester] = joblib.load(os.path.join(semester_path, "hostel_admission_model.pkl"))
                    encoders[gender][semester] = joblib.load(os.path.join(semester_path, "category_encoder.pkl"))

# Load all models and encoders at startup
load_models()

@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        income = data['income']
        sgpa = data['sgpa']
        district = data['district']
        category = data['category']
        gender = data['gender']
        semester = data['semester']
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    
    # Convert values using mappings
    score_income = get_income_weightage(income)
    score_acad = math.ceil(sgpa * 4)
    score_dist = get_distance_weightage(district)
    
    # Validate gender and semester
    if gender not in models or semester not in models[gender]:
        return jsonify({"error": "Invalid gender or semester"}), 400
    
    model = models[gender][semester]
    category_encoder = encoders[gender][semester]
    
    # Ensure the category is encoded properly
    if category not in ['SC', 'ST', 'OBC', 'GENERAL', 'BPL', 'OEC']:
        category = 'GENERAL'
    
    # Encode the category
    encoded_category = category_encoder.transform([category])[0]
    
    # Calculate total score
    total_score = score_income + score_acad + score_dist
    
    # Prepare input for model prediction
    input_data = [[score_income, score_acad, score_dist, total_score, encoded_category]]
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of approval (Yes)
    
    # Calculate predicted percentage
    predicted_percentage = prediction_prob * 100
    
    # Prepare the response
    response = {
        "total_score": total_score,
        "predicted_percentage": predicted_percentage,
        "approval_prediction": "Yes" if prediction[0] == 1 else "No"
    }
    
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=3000,host='0.0.0.0')