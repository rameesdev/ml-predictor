import joblib
import pandas as pd
import os

def main():
    # Check if model files exist
    if not os.path.exists("hostel_admission_model.pkl") or not os.path.exists("category_encoder.pkl"):
        print("Error: Model files not found. Please ensure the model has been trained first.")
        return
    
    try:
        # Load the model and encoder
        pipeline = joblib.load("hostel_admission_model.pkl")
        label_encoder = joblib.load("category_encoder.pkl")
        
        # Get input from user
        print("\n=== Hostel Admission Probability Calculator ===")
        print("Please enter the following information:")
        
        # Collect inputs with validation
        try:
            score_income = float(input("Income Score (0-100, higher score = lower income): "))
            if not 0 <= score_income <= 100:
                raise ValueError("Income score must be between 0 and 100")
            
            score_dist = float(input("Distance Score (0-30, higher score = farther distance): "))
            if not 0 <= score_dist <= 30:
                raise ValueError("Distance score must be between 0 and 30")
            
            keam_rank = int(input("KEAM Rank: "))
            if keam_rank <= 0:
                raise ValueError("KEAM Rank must be greater than 0")
            
            print("\nPlease select a category from the following options:")
            print("1: GENERAL")
            print("2: BPL (Below Poverty Line)")
            print("3: SC (Scheduled Caste)")
            print("4: ST (Scheduled Tribe)")
            print("5: OBC (Other Backward Classes)")
            print("6: OEC (Other Eligible Communities)")
            print("7: BH (Backward Hindu)")
            
            category_option = int(input("\nEnter the number corresponding to your category: "))
            
            # Map option to category name
            category_mapping = {
                1: "GENERAL",
                2: "BPL",
                3: "SC",
                4: "ST",
                5: "OBC",
                6: "OEC",
                7: "BH"
            }
            
            if category_option not in category_mapping:
                raise ValueError("Invalid category option")
                
            category = category_mapping[category_option]
            
        except ValueError as e:
            print(f"Input error: {e}")
            return
        
        # Calculate weighted score (70% income, 30% distance)
        weighted_score = (score_income * 0.7) + (score_dist * 0.3)
        
        # Define category priority
        category_priority = {
            'BPL': 5,
            'SC': 4,
            'ST': 3,
            'OBC': 2,
            'OEC': 1,
            'BH': 1,
            'GENERAL': 0
        }
        
        # Create dataframe for prediction
        input_data = pd.DataFrame({
            'SCORE INCOME': [score_income],
            'SCORE DIST.': [score_dist],
            'WEIGHTED_SCORE': [weighted_score],
            'CATEGORY_ENCODED': [label_encoder.transform([category])[0]],
            'KEAM RANK': [keam_rank],
        })
        
        # Calculate admission priority
        max_rank = 100000  # Use a reasonable maximum rank
        category_priority_value = category_priority.get(category, 0)
        rank_priority = (max_rank - keam_rank) / max_rank
        
       
        
        # Make prediction
        admission_probability = pipeline.predict_proba(input_data)[0][1]
        
        # Display results
        print("\n=== Results ===")
        print(f"Income Score: {score_income}")
        print(f"Distance Score: {score_dist}")
        print(f"Category: {category}")
        print(f"KEAM Rank: {keam_rank}")
        print(f"Weighted Score (70% income, 30% distance): {weighted_score:.2f}")
        print(f"Probability of Admission: {admission_probability:.2%}")
        
        # Give interpretation
        if admission_probability >= 0.8:
            print("\nInterpretation: Very High chance of admission")
        elif admission_probability >= 0.6:
            print("\nInterpretation: High chance of admission")
        elif admission_probability >= 0.4:
            print("\nInterpretation: Moderate chance of admission")
        elif admission_probability >= 0.2:
            print("\nInterpretation: Low chance of admission")
        else:
            print("\nInterpretation: Very Low chance of admission")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()