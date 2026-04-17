Forest Fire Weather Index Prediction System

This project is an end-to-end Machine Learning system that predicts the  Fire Weather Index (FWI) using environmental inputs. It integrates model training, API serving, and optional web interface for real-time predictions.

---------------------------------------------
 Features

-  Predict Fire Weather Index (FWI)
- FastAPI-based backend for real-time inference
- Streamlit UI for interactive predictions
- Modular project structure (scalable & maintainable)

---------------------------------------------

Tech Stack

- Python
- Scikit-learn / ML Model
- FastAPI (Backend API)
- Streamlit (Frontend UI)
---------------------------------------------

Project Structure
FWI-PREDICTION/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schema.py
│   └── model/
│       ├── ridge.pkl
│       └── scalar.pkl
│
├── notebooks/
│   ├── algerian_forest.ipynb
│   └── dataset.csv
│
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
└── venv/

---------------------------------------------
 Machine Learning Model

Problem Statement

Forest fires are heavily influenced by environmental conditions. This project predicts the Fire Weather Index (FWI), which indicates the potential intensity of fire under given conditions.


Input Features

The model takes the following inputs:

- Temperature (°C)
- Relative Humidity (%)
- Wind Speed (km/h)
- Rainfall (mm)

-----------------------------------------------

 Model Workflow

1. Data Collection
   - Historical weather and fire-related data

2. Data Preprocessing
   - Handling missing values
   - Feature scaling / normalization
   - Feature selection

3. Model Training
   - A regression model is trained to predict FWI
   - Common choices include:
     - Linear Regression
     - Random Forest Regressor
     - Gradient Boosting

4. Model Evaluation
   - Metrics used:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R² Score

5. Model Saving


-----------------------------------------

Installation

1.Clone the repository
`
git clone https://github.com/tusharpatidar2006/FWI-prediction/tree/main/app

2. Create virtual environment
python -m venv venv
venv\Scripts\activate 

3. Install dependencies
pip install -r requirements.txt

Run FastAPI Server using URL
https://fwi-prediction-5ca6.onrender.com

Run Streamlit URL
https://fwi-prediction-for-algerian-forest.streamlit.app/