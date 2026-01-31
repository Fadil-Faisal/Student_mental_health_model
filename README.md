# Student Mental Health Predictor
A thoughtful, privacy-forward web tool that uses machine learning to help estimate student anxiety levels and identify students who may benefit from mental health counseling. Built with Python, scikit-learn, and Flask, this project is intended for educational and research use — not a replacement for professional medical advice.

---

✨ Features
- Interactive web UI to input student data and receive predictions.
- Two Random Forest models:
  - Anxiety level prediction
  - Counseling need prediction
- Preprocessing pipeline with label encoders and feature scaling for robust input handling.
- Model training script with evaluation (classification report & confusion matrix).
- Clean project structure with reusable model artifacts (pickled models, scalers, encoders) for quick serving.

---

Table of Contents
- [About](#about)
- [Demo / Quick Preview](#demo--quick-preview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Model & Pipeline](#model--pipeline)
- [Getting Started](#getting-started)
- [Training Models](#training-models)
- [Running the Web App](#running-the-web-app)
- [API Example](#api-example)
- [Evaluation & Results](#evaluation--results)
- [Improvements & Next Steps](#improvements--next-steps)
- [Contributing](#contributing)
- [License & Disclaimer](#license--disclaimer)
- [Acknowledgements](#acknowledgements)

---

About
This project demonstrates how machine learning can be applied to student mental health datasets to infer anxiety and counseling needs from self-reported and demographic attributes (age, gender, academic stressors, sleep quality, etc.). The goal is to provide an approachable research/teaching tool to explore feature importance, model behavior, and simple deployment with Flask.

Demo / Quick Preview
- Launch the Flask app (instructions below), open http://127.0.0.1:5000/ and enter student information via the form to see predictions and confidence indicators.
- The UI displays:
  - Anxiety prediction + confidence bar
  - Counseling-need prediction + confidence bar
  - Notes and recommended next steps (informational only)

Repository Structure
- app.py — Flask application that loads pickled models & serves the UI and prediction API.
- train_model.py — Data preprocessing, model training, evaluation, and serialization (pickle) of models/scalers/encoders.
- setup.py / setup_script.py / something.py — project scaffolding and helper logic used to create directories and template files.
- requirements.txt — pinned Python dependencies for the project.
- data/ — place the dataset CSV here (e.g. `student_mental_health.csv`).
- models/ — serialized artifacts created after training:
  - anxiety_model.pkl
  - counseling_model.pkl
  - scaler_anxiety.pkl
  - scaler_counseling.pkl
  - label_encoders.pkl
- templates/index.html — HTML user interface (Bootstrap-based).
- static/ — CSS, JS, images used by the front-end.

Dataset
- This project uses the "Student Mental Health" dataset (referenced in project scripts). A commonly used copy can be found here:
  - https://www.kaggle.com/datasets/shariful07/student-mental-health
- Place the downloaded CSV file in the `data/` directory. The training scripts expect a CSV path such as `data/student_mental_health.csv` (or `data/testing_data1.csv` depending on the script configuration).

Model & Pipeline (high-level)
1. Load CSV data and perform basic cleaning (drop NA rows).
2. Encode categorical features with LabelEncoder and persist encoders to `models/label_encoders.pkl`.
3. Split feature sets for:
   - Anxiety prediction (targets such as `AnxietyLevel` / `StressLevel` as applicable)
   - Counseling need prediction (`NeedCounseling`)
4. Standardize numeric features with StandardScaler; two scalers are saved for serving.
5. Train RandomForestClassifier models (n_estimators=100, random_state=42).
6. Evaluate using `classification_report` and `confusion_matrix` (printed during training).
7. Save trained models and scalers to `models/`.

Getting Started (local)
Prerequisites:
- Python 3.8+ recommended
- Git (optional)
- Virtual environment recommended

Quick setup:
```bash
# Clone the repo (if not already)
git clone https://github.com/Fadil-Faisal/Student_mental_health_model.git
cd Student_mental_health_model

# Create and activate a virtual environment
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Training Models
1. Ensure the dataset CSV is in `data/` (update paths in `train_model.py` if necessary).
2. Train and evaluate:
```bash
python train_model.py
```
- This will run preprocessing, train both Random Forest models, print classification reports & confusion matrices, and save model artifacts to `models/`.

Running the Web App
After training (or if you have saved model artifacts), start the Flask app:
```bash
python app.py
```
Open http://127.0.0.1:5000/ in your browser to access the interactive UI.

API Example (programmatic)
The Flask app exposes endpoints used by the UI. Example prediction request (update field names to match the project form):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 20,
    "Gender": "Female",
    "SleepQuality": 3,
    "AcademicPressure": 4,
    "SocialSupport": 2,
    "...": "..." 
  }'
```
Response format (example):
```json
{
  "success": true,
  "anxiety_prediction": "Low/Medium/High",
  "anxiety_confidence": 0.92,
  "counseling_prediction": "Yes/No",
  "counseling_confidence": 0.78
}
```
Note: Confirm exact input field names by inspecting `templates/index.html` or the prediction route in `app.py`.

Evaluation & Results
- `train_model.py` prints evaluation metrics (precision, recall, f1-score) and confusion matrices for each target at the end of training. Check those outputs to gauge model performance on your dataset.
- Metrics depend on dataset quality and label distribution. Consider stratified sampling, cross-validation, and additional metrics if class imbalance is significant.

Improvements & Next Steps
- More robust preprocessing: handle missing values, outliers, and finer feature engineering (text responses, multi-select inputs).
- Try other models (XGBoost, LightGBM) and compare with cross-validation.
- Calibrate probabilities (e.g., isotonic or Platt scaling) for more reliable confidence scores.
- Add input validation and secure endpoints for production.
- Add unit tests for preprocessing and model inference to ensure reproducibility.
- Consider privacy-focused deployment: anonymization, encryption in transit & at rest, and access controls.

Contributing
Contributions are welcome! Suggested workflow:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Add tests (if applicable) and update docs.
4. Open a pull request describing the changes and motivation.

When contributing model experiments or data processing changes, include:
- The dataset/variant used (or subset)
- The preprocessing steps
- Evaluation summaries or notebooks

License & Disclaimer
- License: Add an appropriate license file (MIT is a common choice). If you want, this project recommends the MIT license.
- Medical disclaimer: This tool is for educational/research purposes only and is NOT a medical device. It is not a substitute for professional mental health assessment and care. Always recommend users seek professional evaluation if they are in distress.

Suggested license snippet to include in repository (create LICENSE):
```
MIT License

Copyright (c) 2026 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

Acknowledgements
- Dataset: shariful07's Student Mental Health dataset (Kaggle)
- Front-end inspired by Bootstrap UI patterns for simple, accessible forms.

Contact & Support
For questions, feature requests, or collaboration:
- GitHub: https://github.com/Fadil-Faisal/Student_mental_health_model
- (Add email or other contact info if desired)

---

Thank you for building this thoughtful project. If you'd like, I can:
- Create the README.md in the repository directly,
- Add badges (CI / license),
- Generate a short contribution guideline or CODE_OF_CONDUCT,
- Or produce a concise developer guide explaining the main functions in train_model.py and app.py.
Which would you like next?
