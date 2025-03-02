from flask import Flask, request, jsonify, render_template
import joblib
import os
import math
import pandas as pd

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
                    model_file = os.path.join(semester_path, "hostel_admission_model.pkl")
                    encoder_file = os.path.join(semester_path, "category_encoder.pkl")
                    if os.path.exists(model_file) and os.path.exists(encoder_file):
                        models[gender][semester] = joblib.load(model_file)
                        encoders[gender][semester] = joblib.load(encoder_file)

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
        # Check if semester is specified
        if 'semester' not in data:
            return jsonify({"error": "Missing semester parameter"}), 400
            
        semester = data['semester']
        gender = data.get('gender', 'male')  # Default to male if not specified
        
        # For S1 semester with 'male' gender, use the specialized function
        if semester == 'S1':
            return predict_s1(data)
        # For all other semesters, use the original function
        else:
            return predict_other_semesters(data)
            
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Specialized prediction for S1 semester
def predict_s1(data):
    try:
        # Set paths for the S1 male model based on actual structure
        if data["gender"] =="male":
                          model_path = os.path.join(MODEL_DIR, "male", "S1", "hostel_admission_model.pkl")
                          encoder_path = os.path.join(MODEL_DIR, "male", "S1", "category_encoder.pkl")
        else:
                          model_path = os.path.join(MODEL_DIR, "female", "S1", "hostel_admission_model.pkl")
                          encoder_path = os.path.join(MODEL_DIR, "female", "S1", "category_encoder.pkl")
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            return jsonify({"error": "S1 model files not found"}), 500
        
        # Load the S1 model and encoder
        pipeline = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        # Extract S1-specific fields
        try:
            score_income = float(data['income'])
            score_income = (get_income_weightage(score_income)*70)/40
            print(score_income)
            score_dist = str(data['district'])
            score_dist = (get_distance_weightage(score_dist)*30)/20
            print(score_dist)
            keam_rank = int(data['keam_rank'])
            category = data['category']
            
        except KeyError as e:
            return jsonify({"error": f"Missing required S1 parameter: {str(e)}"}), 400
        except ValueError:
            return jsonify({"error": "Invalid input format for S1 data"}), 400
        
        # Calculate weighted score (70% income, 30% distance)
        weighted_score = (score_income * 0.7) + (score_dist * 0.3)
        total_score = score_income+score_dist
        
        # Define category priority
        category_priority = {
            'BPL': 5, 'SC': 4, 'ST': 3, 'OBC': 2, 
            'OEC': 1, 'BH': 1, 'GENERAL': 0
        }
        
        try:
            # Transform category
            encoded_category = label_encoder.transform([category])[0]
            
            # Create dataframe for S1 prediction
            input_data = pd.DataFrame({
                'SCORE INCOME': [score_income],
                'SCORE DIST.': [score_dist],
                'WEIGHTED_SCORE': [weighted_score],
                'CATEGORY_ENCODED': [encoded_category],
                'KEAM RANK': [keam_rank]
            })
            
            # Calculate admission priority
            max_rank = 100000
            category_priority_value = category_priority.get(category, 0)
            rank_priority = (max_rank - keam_rank) / max_rank
            
            
            # Make prediction
            admission_probability = pipeline.predict_proba(input_data)[0][1]
            prediction_result = 1 if admission_probability > 0.5 else 0
            predicted_percentage = admission_probability * 100
            
            # Format response
            response = {
                "total_score": round(total_score, 1),
                "predicted_percentage": round(predicted_percentage, 1),
                "approval_prediction": "Yes" if prediction_result == 1 else "No"
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": f"S1 prediction error: {str(e)}"}), 500
    
    except Exception as e:
        return jsonify({"error": f"S1 processing error: {str(e)}"}), 500

# Original prediction for other semesters
def predict_other_semesters(data):
    try:
        # Extract required fields for other semesters
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
        return jsonify({"error": f"Invalid gender or semester: {gender}, {semester}"}), 400
    
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
    app.run(debug=True, port=3000, host='0.0.0.0')