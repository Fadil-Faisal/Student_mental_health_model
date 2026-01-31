from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load the saved models and preprocessors
@app.before_request
def load_models():
    global anxiety_model, counseling_model, scaler_anxiety, scaler_counseling, label_encoders, feature_columns
    
    try:
        with open('models/anxiety_model.pkl', 'rb') as f:
            anxiety_model = pickle.load(f)
        
        with open('models/counseling_model.pkl', 'rb') as f:
            counseling_model = pickle.load(f)
        
        with open('models/scaler_anxiety.pkl', 'rb') as f:
            scaler_anxiety = pickle.load(f)
        
        with open('models/scaler_counseling.pkl', 'rb') as f:
            scaler_counseling = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
            
        # Load feature columns to ensure consistency
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
        print("Models loaded successfully")
        print("Anxiety model features:", feature_columns['anxiety'])
        print("Counseling model features:", feature_columns['counseling'])
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_model.py first to generate the model files.")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Map form field names to model column names and ensure all required columns are present
def prepare_input_data(form_data):
    # Current timestamp for the Timestamp field
    current_timestamp = datetime.now().strftime('%d-%m-%Y %H:%M')
    
    # Map MaritalStatus to a numeric Social Interaction value
    social_interaction_map = {
        'Single': 5,
        'Married': 7,
        'Divorced': 3
    }
    social_interaction = social_interaction_map.get(form_data.get('MaritalStatus', 'Single'), 5)
    
    # Define mappings and default values with all numeric values
    mapped_data = {
        'Timestamp': current_timestamp,
        'Age': int(form_data.get('Age', 25)),
        'Gender': form_data.get('Gender', 'Female'),
        'Academic Year': int(form_data.get('YearOfStudy', 1)),
        'Course of Study': form_data.get('CourseLevel', 'Business'),
        'Sleep Duration': float(form_data.get('SleepQuality', 7)),
        'Stress Level': int(form_data.get('StressLevel', 5)),
        'Social Interaction': social_interaction,  # Using numeric value
        'Physical Activity': 3,  # Default value
        'Family Support': 5,     # Default value
        'Previous Mental Health Issues': 'No',  # Default value
        'Study Hours per Week': 25,  # Default value
        'Exam Stress': 5,        # Default value
        'Academic Performance (GPA)': float(form_data.get('CGPA', 3.0)),
        'Self-reported Anxiety Symptoms': int(form_data.get('AnxietyLevel', 5)),
        'Depression Indicators': int(form_data.get('DepressionLevel', 5)),
        'Coping Mechanisms': 'Average',  # Default value
        # Note: We're NOT including 'AnxietyLevel' and 'NeedCounseling' here
        # as they are the target variables we're trying to predict
    }
    
    return mapped_data

# Handle label encoding for inputs, including unseen labels
def handle_label_encoding(df, label_encoders):
    for col in df.columns:
        if col in label_encoders:
            # Create a copy of the column with values that can be encoded
            encodable_values = []
            for val in df[col]:
                # Check if the value is in the encoder's classes
                if val not in label_encoders[col].classes_:
                    # If not, use the first class as a fallback
                    encodable_values.append(label_encoders[col].classes_[0])
                else:
                    encodable_values.append(val)
            
            # Transform the encodable values
            df[col] = label_encoders[col].transform(encodable_values)
    
    return df

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Print received form data for debugging
        print("Received form data:", form_data)
        
        # Prepare input data with all required columns and proper mapping
        mapped_data = prepare_input_data(form_data)
        
        # Print mapped data for debugging
        print("Mapped data:", mapped_data)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([mapped_data])
        
        # Handle label encoding for categorical features
        input_df = handle_label_encoding(input_df, label_encoders)
        
        # Convert all columns to numeric where possible
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='raise')
            except:
                pass
        
        # For anxiety prediction, make sure we use only the features that were used during training
        anxiety_features = pd.DataFrame()
        for col in feature_columns['anxiety']:
            if col in input_df.columns:
                anxiety_features[col] = input_df[col]
            else:
                # Set default value for missing columns
                anxiety_features[col] = 0
        
        # For counseling need prediction, make sure we use only the features that were used during training
        counseling_features = pd.DataFrame()
        for col in feature_columns['counseling']:
            if col in input_df.columns:
                counseling_features[col] = input_df[col]
            else:
                # Set default value for missing columns
                counseling_features[col] = 0
        
        # Print feature columns for debugging
        print("Anxiety features:", anxiety_features.columns.tolist())
        print("Expected anxiety features:", feature_columns['anxiety'])
        
        # Scale the features
        anxiety_features_scaled = scaler_anxiety.transform(anxiety_features)
        counseling_features_scaled = scaler_counseling.transform(counseling_features)
        
        # Make predictions
        anxiety_prediction = anxiety_model.predict(anxiety_features_scaled)[0]
        anxiety_probability = np.max(anxiety_model.predict_proba(anxiety_features_scaled)[0]) * 100
        
        counseling_prediction = counseling_model.predict(counseling_features_scaled)[0]
        counseling_probability = np.max(counseling_model.predict_proba(counseling_features_scaled)[0]) * 100
        
        # Map numerical predictions back to categorical labels
        anxiety_label = "High" if anxiety_prediction > 5 else "Low"
        counseling_label = "Recommended" if counseling_prediction > 0 else "Not Needed"
        
        # Return predictions
        return jsonify({
            'success': True,
            'anxiety_level': anxiety_label,
            'anxiety_confidence': f"{anxiety_probability:.2f}%",
            'counseling_need': counseling_label,
            'counseling_confidence': f"{counseling_probability:.2f}%"
        })
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback_str
        })

# Add a diagnosis endpoint for more detailed analysis
@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Prepare input data with all required columns and proper mapping
        mapped_data = prepare_input_data(form_data)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([mapped_data])
        
        # Handle label encoding for categorical features
        input_df = handle_label_encoding(input_df, label_encoders)
        
        return jsonify({
            'success': True,
            'message': 'Detailed diagnosis would be provided here',
            'important_factors': ['Sleep Quality', 'Academic Pressure', 'Social Support']
        })
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback_str
        })

if __name__ == '__main__':
    app.run(debug=True)