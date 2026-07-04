# ✈️ FlightFair: ML-Driven Predictive Pricing Engine

Predicting airline ticket prices using Machine Learning and feature engineering.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![License](https://img.shields.io/badge/License-MIT-red)

---

## 📌 Project Overview

Flight ticket prices fluctuate due to multiple factors such as airline, route, departure time, arrival time, journey duration, total stops, and travel season. Estimating these prices manually is difficult because of the complex relationships between these variables.

This project builds a complete Machine Learning pipeline that predicts airline ticket prices using historical flight data. The workflow includes data preprocessing, feature engineering, exploratory data analysis, model training, hyperparameter tuning, and performance evaluation using multiple regression algorithms.

---

## 🚀 Features

* Comprehensive data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering from date and time attributes
* Handling categorical variables through encoding
* Outlier analysis
* Multiple regression model comparison
* Hyperparameter tuning
* XGBoost-based final prediction model
* Model serialization using Pickle

---

## 📂 Project Structure

```text
FlightFair/
│
├── data/
│   ├── Train.xlsx
│   └── Test.xlsx
│
├── notebooks/
│   └── Flight_Flair_Prediction.ipynb
│
├── models/
│   └── xgboost_model.pkl
│
├── images/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset Features

The model learns from several flight-related attributes including:

* Airline
* Source Airport
* Destination Airport
* Date of Journey
* Departure Time
* Arrival Time
* Duration
* Total Stops
* Additional Information

Target Variable:

* **Price**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Pickle

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

* Handling missing values
* Parsing date and time columns
* Converting categorical variables into numerical representations
* Removing unnecessary features

### 2. Exploratory Data Analysis

* Distribution analysis
* Feature relationships
* Correlation analysis
* Outlier detection

### 3. Feature Engineering

* Journey day and month extraction
* Departure hour and minute
* Arrival hour and minute
* Journey duration conversion
* Route-based feature creation

### 4. Model Building

Different regression models were explored before selecting the final model.

Examples include:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor

### 5. Hyperparameter Optimization

Grid Search / Randomized Search techniques were used to improve model performance by selecting optimal parameters.

### 6. Model Evaluation

Performance was evaluated using regression metrics such as:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## ▶️ Getting Started

### Clone the repository

```bash
git clone https://github.com/abhi-bhavsar/FlightFair-ML-Driven-Predictive-Pricing-Engine.git
cd FlightFair-ML-Driven-Predictive-Pricing-Engine
```

### Create a virtual environment using uv

```bash
uv venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
uv pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Flight_Flair_Prediction.ipynb
```

---

## 📈 Future Improvements

* Flask or Streamlit web application
* REST API for predictions
* Model versioning
* Automated training pipeline
* CI/CD deployment
* Docker support
* Cloud deployment

---

## 👨‍💻 Author

**Abhishek Bhavsar**

B.Tech Computer Science (Data Science)

Machine Learning • Generative AI • RAG • Agentic AI

GitHub: https://github.com/abhi-bhavsar

---

## ⭐ If you found this repository useful, consider giving it a star!
