import os
import shutil

def create_project_structure():
    # Create directories
    directories = [
        'static',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'models',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create train_model.py
    with open('train_model.py', 'w') as f:
        f.write('''
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
    
    # Save the label encoders for later use in the Flask app
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return df

# Function to train models
def train_models(df):
    # Prepare features for anxiety/stress prediction
    anxiety_features = df.drop(['AnxietyLevel', 'StressLevel', 'DepressionLevel', 'SleepQuality', 'NeedCounseling'], axis=1)
    anxiety_target = df['AnxietyLevel']
    
    # Prepare features for counseling need prediction
    counseling_features = df.drop(['NeedCounseling'], axis=1)
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
    
    print("\\nCounseling Need Prediction Results:")
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
    dataset_path = 'data/student_mental_health.csv'  # Update this path
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download the dataset from https://www.kaggle.com/datasets/shariful07/student-mental-health")
        print("and place it in the data directory or update the path in the code.")
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
''')
    print("Created train_model.py")

    # Create app.py
    with open('app.py', 'w') as f:
        f.write('''
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the saved models and preprocessors
@app.before_first_request
def load_models():
    global anxiety_model, counseling_model, scaler_anxiety, scaler_counseling, label_encoders
    
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
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_model.py first to generate the model files.")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert form data to DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Preprocess input data
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Convert to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='ignore')
        
        # Make predictions
        # For anxiety prediction
        anxiety_features = input_df.drop(['AnxietyLevel', 'StressLevel', 'DepressionLevel', 'SleepQuality', 'NeedCounseling'], axis=1, errors='ignore')
        anxiety_features_scaled = scaler_anxiety.transform(anxiety_features)
        anxiety_prediction = anxiety_model.predict(anxiety_features_scaled)[0]
        anxiety_probability = np.max(anxiety_model.predict_proba(anxiety_features_scaled)[0]) * 100
        
        # For counseling need prediction
        counseling_features = input_df.copy()
        if 'NeedCounseling' in counseling_features.columns:
            counseling_features = counseling_features.drop(['NeedCounseling'], axis=1)
        counseling_features_scaled = scaler_counseling.transform(counseling_features)
        counseling_prediction = counseling_model.predict(counseling_features_scaled)[0]
        counseling_probability = np.max(counseling_model.predict_proba(counseling_features_scaled)[0]) * 100
        
        # Map numerical predictions back to categorical labels
        anxiety_label = "High" if anxiety_prediction > 0.5 else "Low"
        counseling_label = "Yes" if counseling_prediction > 0.5 else "No"
        
        # Return predictions
        return jsonify({
            'success': True,
            'anxiety_level': anxiety_label,
            'anxiety_confidence': f"{anxiety_probability:.2f}%",
            'counseling_need': counseling_label,
            'counseling_confidence': f"{counseling_probability:.2f}%"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add a diagnosis endpoint for more detailed analysis
@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert form data to DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Preprocess input data (similar to predict route)
        # ...
        
        # Return a more detailed analysis
        # This could include feature importance, recommendations, etc.
        # For demonstration, we'll return a simple response
        return jsonify({
            'success': True,
            'message': 'Detailed diagnosis would be provided here',
            'important_factors': ['Sleep Quality', 'Academic Pressure', 'Social Support']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
''')
    print("Created app.py")

    # Create index.html in templates directory
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Mental Health Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .prediction-card {
            display: none;
            margin-top: 20px;
        }
        .form-container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .result-container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #343a40;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student Mental Health Prediction</h1>
            <p class="lead">This tool uses machine learning to predict anxiety levels and counseling needs based on student data</p>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="form-container">
                    <h2>Enter Student Information</h2>
                    <form id="prediction-form">
                        <div class="mb-3">
                            <label for="Age" class="form-label">Age</label>
                            <input type="number" class="form-control" id="Age" name="Age" required>
                        </div>
                        
                        <div class="mb-3">
                            <label for="Gender" class="form-label">Gender</label>
                            <select class="form-control" id="Gender" name="Gender" required>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="CourseLevel" class="form-label">Course Level</label>
                            <select class="form-control" id="CourseLevel" name="CourseLevel" required>
                                <option value="Undergraduate">Undergraduate</option>
                                <option value="Graduate">Graduate</option>
                                <option value="PhD">PhD</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="YearOfStudy" class="form-label">Year of Study</label>
                            <input type="number" class="form-control" id="YearOfStudy" name="YearOfStudy" min="1" max="7" required>
                        </div>
                        
                        <div class="mb-3">
                            <label for="CGPA" class="form-label">CGPA</label>
                            <input type="number" step="0.01" class="form-control" id="CGPA" name="CGPA" min="0" max="4" required>
                        </div>
                        
                        <div class="mb-3">
                            <label for="MaritalStatus" class="form-label">Marital Status</label>
                            <select class="form-control" id="MaritalStatus" name="MaritalStatus" required>
                                <option value="Single">Single</option>
                                <option value="Married">Married</option>
                                <option value="Divorced">Divorced</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="Depression" class="form-label">Depression Level (1-10)</label>
                            <input type="range" class="form-range" id="Depression" name="DepressionLevel" min="1" max="10" required>
                            <div class="d-flex justify-content-between">
                                <span>Low</span>
                                <span>High</span>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="Anxiety" class="form-label">Anxiety Level (1-10)</label>
                            <input type="range" class="form-range" id="Anxiety" name="AnxietyLevel" min="1" max="10" required>
                            <div class="d-flex justify-content-between">
                                <span>Low</span>
                                <span>High</span>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="Stress" class="form-label">Stress Level (1-10)</label>
                            <input type="range" class="form-range" id="Stress" name="StressLevel" min="1" max="10" required>
                            <div class="d-flex justify-content-between">
                                <span>Low</span>
                                <span>High</span>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="Sleep" class="form-label">Sleep Quality (1-10)</label>
                            <input type="range" class="form-range" id="Sleep" name="SleepQuality" min="1" max="10" required>
                            <div class="d-flex justify-content-between">
                                <span>Poor</span>
                                <span>Excellent</span>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary w-100">Predict</button>
                    </form>
                </div>
            </div>
            
            <div class="col-md-6">
                <div id="results" class="result-container" style="display: none;">
                    <h2>Prediction Results</h2>
                    <div class="alert alert-info">
                        <p>These predictions are based on the machine learning model trained on student mental health data:</p>
                    </div>
                    
                    <div class="card mb-3">
                        <div class="card-header">Anxiety Level Prediction</div>
                        <div class="card-body">
                            <h3 id="anxiety-result" class="card-title"></h3>
                            <p id="anxiety-confidence" class="card-text"></p>
                            <div class="progress mb-3">
                                <div id="anxiety-progress" class="progress-bar bg-warning" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Counseling Need Prediction</div>
                        <div class="card-body">
                            <h3 id="counseling-result" class="card-title"></h3>
                            <p id="counseling-confidence" class="card-text"></p>
                            <div class="progress mb-3">
                                <div id="counseling-progress" class="progress-bar bg-info" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="alert alert-warning mt-3">
                        <strong>Note:</strong> This is a predictive tool and should not replace professional assessment. If you're experiencing mental health challenges, please consult with a healthcare provider.
                    </div>
                </div>
                
                <div id="error-message" class="alert alert-danger mt-3" style="display: none;">
                    An error occurred during prediction. Please try again.
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            // Handle form submission
            $("#prediction-form").submit(function(e) {
                e.preventDefault();
                
                // Show loading state
                $("button[type='submit']").prop("disabled", true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
                
                // Send prediction request
                $.ajax({
                    url: "/predict",
                    type: "POST",
                    data: $(this).serialize(),
                    success: function(response) {
                        if (response.success) {
                            // Display results
                            $("#anxiety-result").text(response.anxiety_level);
                            $("#anxiety-confidence").text("Confidence: " + response.anxiety_confidence);
                            
                            $("#counseling-result").text(response.counseling_need);
                            $("#counseling-confidence").text("Confidence: " + response.counseling_confidence);
                            
                            // Set progress bars
                            let anxietyConfidence = parseFloat(response.anxiety_confidence);
                            $("#anxiety-progress").css("width", anxietyConfidence + "%");
                            
                            let counselingConfidence = parseFloat(response.counseling_confidence);
                            $("#counseling-progress").css("width", counselingConfidence + "%");
                            
                            // Show results
                            $("#results").show();
                            $("#error-message").hide();
                        } else {
                            // Show error
                            $("#error-message").text(response.error).show();
                            $("#results").hide();
                        }
                    },
                    error: function() {
                        $("#error-message").text("Server error. Please try again later.").show();
                        $("#results").hide();
                    },
                    complete: function() {
                        // Reset button state
                        $("button[type='submit']").prop("disabled", false).text("Predict");
                    }
                });
            });
        });
    </script>
</body>
</html>''')
    print("Created templates/index.html")

    # Create CSS file
    with open('static/css/styles.css', 'w') as f:
        f.write('''
body {
    padding: 20px;
    background-color: #f8f9fa;
    font-family: 'Arial', sans-serif;
}

.prediction-card {
    display: none;
    margin-top: 20px;
}

.form-container {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.result-container {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h1, h2, h3 {
    color: #343a40;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.form-label {
    font-weight: 500;
}

.progress {
    height: 25px;
}

.progress-bar {
    font-weight: bold;
}
''')
    print("Created static/css/styles.css")

    # Create JavaScript file
    with open('static/js/app.js', 'w') as f:
        f.write('''
$(document).ready(function() {
    // Handle form submission
    $("#prediction-form").submit(function(e) {
        e.preventDefault();
        
        // Show loading state
        $("button[type='submit']").prop("disabled", true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
        
        // Send prediction request
        $.ajax({
            url: "/predict",
            type: "POST",
            data: $(this).serialize(),
            success: function(response) {
                if (response.success) {
                    // Display results
                    $("#anxiety-result").text(response.anxiety_level);
                    $("#anxiety-confidence").text("Confidence: " + response.anxiety_confidence);
                    
                    $("#counseling-result").text(response.counseling_need);
                    $("#counseling-confidence").text("Confidence: " + response.counseling_confidence);
                    
                    // Set progress bars
                    let anxietyConfidence = parseFloat(response.anxiety_confidence);
                    $("#anxiety-progress").css("width", anxietyConfidence + "%");
                    
                    let counselingConfidence = parseFloat(response.counseling_confidence);
                    $("#counseling-progress").css("width", counselingConfidence + "%");
                    
                    // Show results
                    $("#results").show();
                    $("#error-message").hide();
                } else {
                    // Show error
                    $("#error-message").text(response.error).show();
                    $("#results").hide();
                }
            },
            error: function() {
                $("#error-message").text("Server error. Please try again later.").show();
                $("#results").hide();
            },
            complete: function() {
                // Reset button state
                $("button[type='submit']").prop("disabled", false).text("Predict");
            }
        });
    });
});
''')
    print("Created static/js/app.js")

    # Create README.md
    with open('README.md', 'w') as f:
        f.write('''# Student Mental Health Prediction Tool

This application uses machine learning to predict student stress/anxiety levels and their need for mental health support based on various factors.

## Project Structure

```
student-mental-health-prediction/
│
├── app.py                   # Flask application
├── train_model.py           # Script to train and save the models
├── setup.py                 # Script to set up project structure
├── README.md                # Project documentation
│
├── data/                    # Store datasets
│   └── student_mental_health.csv
│
├── models/                  # Store trained models and preprocessors
│   ├── anxiety_model.pkl
│   ├── counseling_model.pkl
│   ├── scaler_anxiety.pkl
│   ├── scaler_counseling.pkl
│   └── label_encoders.pkl
│
├── static/                  # Static assets
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   └── images/
│
└── templates/               # HTML templates
    └── index.html
```

## Setup Instructions

1. Clone the repository or download the project files.

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install required packages:
   ```
   pip install flask pandas numpy scikit-learn
   ```

4. Download the dataset from Kaggle:
   [Student Mental Health Dataset](https://www.kaggle.com/datasets/shariful07/student-mental-health)
   
   Place the downloaded CSV file in the `data/` directory.

5. Run the setup script to create the project structure:
   ```
   python setup.py
   ```

6. Train the models:
   ```
   python train_model.py
   ```

7. Start the Flask application:
   ```
   python app.py
   ```

8. Access the application in your web browser at:
   ```
   http://127.0.0.1:5000/
   ```

## How It Works

1. The application uses Random Forest classifiers to predict:
   - Student anxiety levels
   - Need for mental health counseling

2. The prediction is based on various student attributes like age, gender, academic performance, and self-reported mental health indicators.

3. The web interface allows users to input student information and view prediction results with confidence scores.

## Notes

- This tool is for educational purposes only and should not replace professional medical advice.
- The model's accuracy depends on the quality and representativeness of the training data.
- For production use, additional security measures and error handling would be needed.
''')
    print("Created README.md")

    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('''Flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0
Werkzeug==2.0.1
Jinja2==3.0.1
itsdangerous==2.0.1
MarkupSafe==2.0.1
joblib==1.1.0
threadpoolctl==3.0.0
scipy==1.7.1
''')
    print("Created requirements.txt")

    # Create setup.py itself
    with open('setup.py', 'w') as f:
        f.write('''
import os
import shutil
from setup_script import create_project_structure

if __name__ == "__main__":
    print("Setting up Student Mental Health Prediction project...")
    create_project_structure()
    print("\\nSetup complete! Next steps:")
    print("1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/shariful07/student-mental-health")
    print("2. Place the CSV file in the 'data/' directory as student_mental_health.csv")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Train the models: python train_model.py")
    print("5. Run the application: python app.py")
''')
    print("Created setup.py")
    
    # Save this script as setup_script.py
    with open('setup_script.py', 'w') as f:
        f.write(open(__file__).read())
    print("Created setup_script.py")
    
    print("\nProject structure setup complete!")

if __name__ == "__main__":
    create_project_structure()