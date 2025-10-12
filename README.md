# 🇵🇰 Pakistan House Price Predictor

<img src="static/images/logo1.png" alt="Pakistan House Price Predictor Logo" width="250"/>

A **machine learning web app** that predicts house prices across **5 major cities of Pakistan**.  
It uses features like **location, property type, area, bedrooms, and bathrooms** to give an estimated price.  
The model achieves an impressive **R² score of 99.9%**, showing how accurate the predictions are.

---

## 🏠 Project Overview

This project uses an **XGBoost regression model** trained on real housing data from major cities in Pakistan.  
Users just need to select the **city**, **location**, **purpose (sale/rent)**, **property type**, and enter basic details — and the app instantly shows an estimated house price.

---

# 📷 Screenshots

### Frontend Form  
![Pakistan House Pridictor Predictor Form](static/css/images/logo1.png)  



## 🌐 Live Demo

👉 **Youtube Video Here:** [Pakistan House Price Predictor](https://pakistan-homes-price-prediction-1.onrender.com/)

---

## 📊 Model Information

- **Algorithm:** XGBoost Regressor  
- **Data Preprocessing:**  
  - `OneHotEncoder` → For categorical columns  
  - `StandardScaler` → For numerical columns  
- **Target Variable:** `price`

---

## 🏙️ Supported Cities

The app currently supports house price predictions for:

- 🏡 Lahore  
- 🕌 Islamabad  
- 🌆 Karachi  
- 🏢 Faisalabad  
- 🛣️ Rawalpindi  

---

## 🔍 Input Features

| Feature         | Type        | Description |
|-----------------|-------------|-------------|
| `purpose`       | Categorical | For Sale / For Rent |
| `city`          | Categorical | One of the 5 major cities |
| `location`      | Categorical | Specific area within city |
| `property_type` | Categorical | House, Flat, or other |
| `bedrooms`      | Numeric     | Number of bedrooms |
| `bathrooms`     | Numeric     | Number of bathrooms |
| `Area_in_Marla` | Numeric     | Total area in Marla |

**Target:**  
- `price_per_marla` (used to calculate final price)

---

## 🧠 How It Works

1. User enters details in the form.  
2. Backend performs preprocessing:
   - Encodes text values (city, location, property type)
   - Scales numeric features
3. Model predicts the **price per marla** using XGBoost.  
4. Final price = `predicted_price_per_marla × area_in_marla`  
5. Result is displayed instantly on the web app.

---

## ⚙️ Installation Guide

### 1. Clone this repository
```bash
git clone https://github.com/aliahmad552/pakistan_homes_price_prediction.git
cd pakistan_homes_price_prediction
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Flask app
```bash
python app.py


Visit 👉 http://localhost:5000
```
 in your browser.

## 👨‍💻 Author

Ali Ahmad
🎓 BS Software Engineering, The Islamia University of Bahawalpur
📧 aliahmaddawana@example.com

⭐ If you like this project, give it a star on GitHub!
