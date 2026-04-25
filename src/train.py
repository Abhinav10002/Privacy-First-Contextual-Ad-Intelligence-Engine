import pandas as pd  
import xgboost as xgb 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def train_real_model(data_dir, model_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()

    imbalance_ratio = num_neg / num_pos
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
    print("Training XGBoost on real-world data....")

    model = xgb.XGBClassifier(
        n_estimators = 100,
        max_depth = 6,
        learning_rate = 0.1,
        scale_pos_weight = imbalance_ratio,
        random_state = 42,
        eval_metric = 'logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n---Model Evaluation---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/ad_model.pkl",'wb') as f:
        pickle.dump(model,f)
    
    with open(f"{model_dir}/model_columns.pkl", 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
        
    print(f"\nModel saved in {model_dir}")

if __name__ == "__main__":
    train_real_model('data/processed', 'models')