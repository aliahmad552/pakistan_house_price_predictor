# 🇵🇰 Pakistan House Price Predictor

A machine learning web application that predicts house prices in 5 major cities of Pakistan using features like location, property type, area, bedrooms, and bathrooms. The model achieves an **R² score of 99.9%**, making it highly accurate for price estimation.
logo.png)

## 🏠 Project Overview

This project leverages an XGBoost regression model trained on curated housing data from five major Pakistani cities. Users can select the city, location, purpose, property type, and other features, and receive a near-accurate price prediction instantly.

## 🚀 Live Demo

🌐 [Rendered App Link](https://your-render-link.com)

## 📊 Model Performance

- **Model:** XGBoost Regressor  
- **R² Score:** 99.9%  
- **Preprocessing:** OneHotEncoding for categorical features and StandardScaler for numerical features  
- **Target Variable:** `price`

## 🏙️ Supported Cities & Features

The model is trained on housing data from the following Pakistani cities:

- Lahore
- Islamabad
- Karachi
- Faisalabad

### Input Features:

| Feature         | Type        | Description |
|----------------|-------------|-------------|
| `purpose`       | Categorical | For Sale / For Rent |
| `city`          | Categorical | One of the 5 major cities |
| `location`      | Categorical | Specific location within selected city |
| `property_type` | Categorical | Type of property (e.g., House, Flat) |
| `bedrooms`      | Numeric     | Number of bedrooms |
| `bathrooms`     | Numeric     | Number of bathrooms |
| `Area_in_Marla` | Numeric     | Area in Marla |

### Target:

- `price_per_marla`: Used to calculate the final predicted price.

## 🧠 How It Works

1. User fills in the form with house details.
2. Backend processes the data:
   - Encodes categorical variables using `OneHotEncoder`
   - Scales numerical values using `StandardScaler`
   - Predicts `price_per_marla` using XGBoost model
   - Final price is calculated as `predicted_price_per_marla × area`
3. Result is returned in seconds.

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pakistan-house-price-predictor.git
cd pakistan-house-price-predictor
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Flask App
bash
Copy
Edit
python app.py
Visit http://localhost:5000 to view the app.

🗃️ File Structure
cpp
Copy
Edit
├── app.py
├── xgb.pkl
├── ohe.pkl
├── scaler.pkl
├── Cleaned_data.csv
├── static/
│   └── logo.png
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
##🧹 Data Preprocessing Highlights
price_per_marla was engineered as price / area for better regression performance.

Removed outliers and cleaned inconsistent location names.

Applied OneHotEncoding on:

location, property_type, city

Used label mapping for purpose (0 = For Sale, 1 = For Rent)

Scaled numeric columns: bedrooms, bathrooms, Area_in_Marla

##👨‍💻 Author
Ali Ahmad
BS Software Engineering, The Islamia University of Bahawalpur
📧 frextarr.552@example.com
