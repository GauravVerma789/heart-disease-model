
# 🫀 Heart Disease Prediction Model

This project is a Machine Learning-based web app that predicts the likelihood of heart disease in a patient using medical input data.

---

## 📌 Features

- ✅ Predicts risk of heart disease using patient health metrics
- ✅ Built using Python and Scikit-learn
- ✅ Simple and clean web interface using Streamlit / Flask / FastAPI
- ✅ Model trained on real medical dataset (e.g., UCI Heart Disease Dataset)

---

## 🧠 Technologies Used

- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn (for EDA)
- Flask / Streamlit / FastAPI (for deployment)
- HTML/CSS (if applicable for frontend)

---
heart-disease-prediction/
├── 📁 app/                    # Application files (Frontend + Backend)
│   ├── app.py                # Main application (Flask/Streamlit/FastAPI)
│   ├── 📁 templates/         # HTML templates (only for Flask)
│   ├── 📁 static/            # CSS, JS, images (for styling)
│   └── 📁 components/        # Reusable components (optional)
│
├── 📁 model/                  # Machine Learning model files
│   ├── heart_disease_model.pkl   # Trained ML model
│   └── train_model.py            # Script to train the model
│
├── 📁 data/                   # Dataset and data processing
│   └── heart.csv             # Original dataset
│
├── 📁 notebooks/              # Jupyter notebooks (for EDA, model testing)
│   └── heart_analysis.ipynb  # Exploratory data analysis notebook
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules


The model takes the following input features:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Max heart rate achieved
- Exercise-induced angina
- ST depression (oldpeak)
- Slope of the ST segment
- Number of major vessels
- Thalassemia

---

