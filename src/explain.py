import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
import os

def explain_model():
    # 1. Load the Pickle model (the standard one we use)
    if not os.path.exists('models/ad_model.pkl'):
        print("❌ Error: models/ad_model.pkl not found. Run train.py first.")
        return
        
    with open('models/ad_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load Data
    X_test = pd.read_csv('data/processed/X_test.csv')
    # We use a smaller sample (100 rows) because Permutation explainer is slightly slower
    sample_data = X_test.head(100)
    
    print("🧠 Calculating SHAP values using the Model-Agnostic Explainer...")
    
    # 3. Use shap.Explainer instead of TreeExplainer
    # This treats the model as a black-box function, avoiding internal parsing errors
    explainer = shap.Explainer(model.predict, sample_data)
    
    # Calculate SHAP values
    shap_values = explainer(sample_data)

    # 4. Generate Summary Plot
    # 
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, show=False)
    
    # 5. Save the report
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    plt.close()
    
    print("✅ Success! SHAP Summary Plot saved to 'reports/shap_summary.png'")

if __name__ == "__main__":
    explain_model()