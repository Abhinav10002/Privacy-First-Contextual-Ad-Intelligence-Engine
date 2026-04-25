import pandas as pd
import pickle
import numpy as np

def rank_ads(context_data, candidate_apps):
    # 1. Load Model and Artifacts
    with open('models/ad_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)

    results = []

    # 2. Process each candidate ad
    for app in candidate_apps:
        # Create a single row for prediction
        input_row = {
            'os_version': context_data['os_version'],
            'is_4G': context_data['is_4G'],
            'hour': context_data['hour'],
            'day_of_week': context_data['day_of_week'],
            'app_code': str(app) # The candidate ad
        }
        
        # 3. Apply Label Encoding (using the encoders from Phase 3)
        try:
            input_row['os_version'] = encoders['os_version'].transform([str(input_row['os_version'])])[0]
            input_row['app_code'] = encoders['app_code'].transform([str(input_row['app_code'])])[0]
        except ValueError:
            # If the app/os is new, we assign a default (0)
            input_row['os_version'] = 0
            input_row['app_code'] = 0

        # Convert to DataFrame and align columns exactly like training
        input_df = pd.DataFrame([input_row])[model_columns]

        # 4. Get Probability of Click (Class 1)
        prob = model.predict_proba(input_df)[0][1]
        results.append({'app_code': app, 'score': prob})

    # 5. Sort by score descending
    ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return ranked_results

if __name__ == "__main__":
    # Simulate a real-time request
    current_context = {
        'os_version': 'latest', # This matches your dataset values
        'is_4G': 1,
        'hour': 18,        # 6 PM
        'day_of_week': 0   # Monday
    }
    
    # These are 5 "Ad IDs" (app_codes) we want to choose from
    my_candidates = [123, 456, 789, 101, 202]
    
    print("🎯 Ranking Ads for the current context...")
    recommendations = rank_ads(current_context, my_candidates)
    
    for i, rec in enumerate(recommendations):
        print(f"Rank {i+1}: Ad {rec['app_code']} (Probability: {rec['score']:.4f})")