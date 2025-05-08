from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('xgb.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load dropdown data
df = pd.read_csv("Cleaned_data.csv")

# Routes
@app.route('/')
def home():
    cities = sorted(df['city'].unique())
    purposes = sorted(df['purpose'].unique())
    property_types = sorted(df['property_type'].unique())
    return render_template('index.html', cities=cities, purposes=purposes, property_types=property_types)

@app.route('/get_locations', methods=['POST'])
def get_locations():
    data = request.get_json()
    selected_city = data.get('city')
    locations = sorted(df[df['city'] == selected_city]['location'].unique())
    return jsonify({'locations': locations})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract user input (no price input)
        purpose = data['purpose']
        city = data['city']
        location = data['location']
        property_type = data['property_type']
        bedrooms = int(data['bedrooms'])
        baths = int(data['bathrooms'])
        area = float(data['Area_in_Marla'])

        # Encode categorical features using OneHotEncoder (all 4)
        categorical_input = [[location, property_type, city, purpose]]
        encoded_cats = ohe.transform(categorical_input)

        # Scale numerical features (excluding price)
        numerical_input = [[baths, bedrooms, area, 0]]  # dummy price=0
        scaled_nums = scaler.transform(numerical_input)
        scaled_nums = scaled_nums[:, :3]

        # Combine features in the same order used during model training
        final_input = np.concatenate([scaled_nums, encoded_cats], axis=None).reshape(1, -1)

        # Predict price_per_marla
        predicted_price_per_marla = model.predict(final_input)[0]

        # Since price was scaled during training, we need to reverse the scaling (if needed)
        # Multiply with area to get actual price (scaled)
        predicted_scaled_price = predicted_price_per_marla * area

        # Reverse the scaling for the price (if required)
        # If you scaled the 'price' during training, apply inverse scaling here
        final_price = round(predicted_scaled_price)

        return jsonify({'predicted_price': final_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
