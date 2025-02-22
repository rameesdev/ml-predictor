from flask import Flask, request, jsonify
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Define model directory
MODEL_DIR = "MODELS"

# Preload all models and encoders
models = {}
encoders = {}

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
print(models)
# Define route for prediction
@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        score_income = data['score_income']
        score_acad = data['score_acad']
        score_dist = data['score_dist']
        category = data['category']
        gender = data['gender']
        semester = data['semester']
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    
    # Validate gender and semester
    if gender not in models or semester not in models[gender]:
        return jsonify({"error": "Invalid gender or semester"}), 400
    
    model = models[gender][semester]
    category_encoder = encoders[gender][semester]
    print(gender)
    print(semester)
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
    app.run(host="0.0.0.0",debug=True,port=3000)
