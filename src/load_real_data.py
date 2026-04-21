import pandas as pd
import os

def prepare_real_data(input_path):
    if not os.makedirs('data', exist_ok=True) and not os.path.exists(input_path):
        print(f" Error: {input_path} not found.")
        return

    print(" Reading WNS Ad Click dataset...")
    
    # 1. Map columns based on your specific file output
    # We use 'impression_time' as our source for time features
    target_col = 'is_click'
    context_cols = ['impression_time', 'app_code', 'os_version', 'is_4G']
    
    try:
        # 2. Load the data (sampling 200k rows for speed)
        df = pd.read_csv(input_path, usecols=[target_col] + context_cols, nrows=200000)
        
        # 3. Feature Engineering: Time Context
        print(" Converting impression_time to features...")
        df['impression_time'] = pd.to_datetime(df['impression_time'])
        df['hour'] = df['impression_time'].dt.hour
        df['day_of_week'] = df['impression_time'].dt.dayofweek
        
        # 4. Clean up
        # Drop the original timestamp string
        df_final = df.drop(columns=['impression_time'])
        
        # Fill any missing values with the most frequent value (mode)
        df_final = df_final.fillna(df_final.mode().iloc[0])
        
        # 5. Save the prepared raw data
        df_final.to_csv('data/raw_ads_data.csv', index=False)
        
        print(f"Success! Prepared {len(df_final)} rows.")
        print(f"Final Contextual Features: {df_final.columns.tolist()}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prepare_real_data('data/real_ads_data.csv')