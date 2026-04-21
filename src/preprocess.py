import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess_real_data(input_path, output_dir):
    df = pd.read_csv(input_path)

    click_rate = df['is_click'].mean() * 100
    print(f"Current Click-Through Rate (CTR) in data: {click_rate:.2f}%")


    cat_cols = ['os_version','app_code']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    

    os.makedirs('models', exist_ok=True)
    with open('models/label_encoders.pkl','wb') as f:
        pickle.dump(encoders,f)
    
    X = df.drop('is_click',axis=1)
    y = df['is_click']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"✅ Preprocessing complete. Files saved in {output_dir}")
    print(f"Training set size: {X_train.shape}")

if __name__ == "__main__":
    preprocess_real_data('data/raw_ads_data.csv', 'data/processed')