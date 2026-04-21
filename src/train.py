import pandas as pd   
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model(data_dir, model_dir):
    