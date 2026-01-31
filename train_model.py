# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Function to load and preprocess data
def preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Data cleaning and preprocessing
    # Handle missing values if any
    df = df.dropna()
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the label encoders for later use in the Flask app
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return df

# Function to train models
def train_models(df):
    # Prepare features for anxiety level prediction
    # IMPORTANT: Don't include AnxietyLevel in features for anxiety prediction
    anxiety_features = df.drop(['AnxietyLevel', 'NeedCounseling'], axis=1)
    anxiety_target = df['AnxietyLevel']
    
    # Prepare features for counseling need prediction
    # IMPORTANT: Don't include NeedCounseling in features for counseling prediction
    counseling_features = df.drop(['AnxietyLevel', 'NeedCounseling'], axis=1)
    counseling_target = df['NeedCounseling']
    
    # Split data into training and testing sets
    X_anxiety_train, X_anxiety_test, y_anxiety_train, y_anxiety_test = train_test_split(
        anxiety_features, anxiety_target, test_size=0.2, random_state=42)
    
    X_counseling_train, X_counseling_test, y_counseling_train, y_counseling_test = train_test_split(
        counseling_features, counseling_target, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler_anxiety = StandardScaler()
    X_anxiety_train_scaled = scaler_anxiety.fit_transform(X_anxiety_train)
    X_anxiety_test_scaled = scaler_anxiety.transform(X_anxiety_test)
    
    scaler_counseling = StandardScaler()
    X_counseling_train_scaled = scaler_counseling.fit_transform(X_counseling_train)
    X_counseling_test_scaled = scaler_counseling.transform(X_counseling_test)
    
    # Save scalers for later use in the Flask app
    with open('models/scaler_anxiety.pkl', 'wb') as f:
        pickle.dump(scaler_anxiety, f)
    
    with open('models/scaler_counseling.pkl', 'wb') as f:
        pickle.dump(scaler_counseling, f)
    
    # Save feature columns to ensure consistency in prediction
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump({
            'anxiety': list(anxiety_features.columns),
            'counseling': list(counseling_features.columns)
        }, f)
    
    # Train anxiety prediction model
    anxiety_model = RandomForestClassifier(n_estimators=100, random_state=42)
    anxiety_model.fit(X_anxiety_train_scaled, y_anxiety_train)
    
    # Train counseling need prediction model
    counseling_model = RandomForestClassifier(n_estimators=100, random_state=42)
    counseling_model.fit(X_counseling_train_scaled, y_counseling_train)
    
    # Evaluate models
    anxiety_preds = anxiety_model.predict(X_anxiety_test_scaled)
    counseling_preds = counseling_model.predict(X_counseling_test_scaled)
    
    print("Anxiety Level Prediction Results:")
    print(classification_report(y_anxiety_test, anxiety_preds))
    print(confusion_matrix(y_anxiety_test, anxiety_preds))
    
    print("\nCounseling Need Prediction Results:")
    print(classification_report(y_counseling_test, counseling_preds))
    print(confusion_matrix(y_counseling_test, counseling_preds))
    
    # Save models
    with open('models/anxiety_model.pkl', 'wb') as f:
        pickle.dump(anxiety_model, f)
    
    with open('models/counseling_model.pkl', 'wb') as f:
        pickle.dump(counseling_model, f)
    
    return anxiety_model, counseling_model, scaler_anxiety, scaler_counseling

# Main function to execute the pipeline
def main():
    # Set the path to your dataset
    dataset_path = 'data/testing_data1.csv'  # Update this path
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download the dataset and place it in the data directory or update the path in the code.")
        return
    
    # Process data
    print("Preprocessing data...")
    df = preprocess_data(dataset_path)
    
    # Train models
    print("Training models...")
    anxiety_model, counseling_model, _, _ = train_models(df)
    
    print("Models trained and saved successfully!")
    
    return df, anxiety_model, counseling_model

if __name__ == "__main__":
    main()