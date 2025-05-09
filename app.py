from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load both models
sale_model = pickle.load(open('xgb.pkl', 'rb'))
rent_model = pickle.load(open('xgb_forRent.pkl', 'rb'))

# Load encoder and scaler
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load data for dropdowns
df = pd.read_csv('Cleaned_data.csv')
cities = sorted(df['city'].unique())
property_types = sorted(df['property_type'].unique())

@app.route('/')
def index():
    return render_template('index.html', cities=cities, property_types=property_types)

@app.route('/get_locations/<city>')
def get_locations(city):
    filtered = df[df['city'] == city]
    locations = sorted(filtered['location'].unique())
    return jsonify({'locations': locations})

@app.route('/', methods=['POST'])
def predict():
    city = request.form['city']
    location = request.form['location']
    property_type = request.form['property_type']
    bedrooms = int(request.form['bedrooms'])
    baths = int(request.form['baths'])
    area = float(request.form['area'])
    model_type = request.form['model_type']  # sale or rent

    # Choose the correct model
    model = sale_model if model_type == 'sale' else rent_model

    # Create input dataframe
    input_df = pd.DataFrame([{
        'city': city,
        'location': location,
        'property_type': property_type,
        'bedrooms': bedrooms,
        'baths': baths,
        'Area_in_Marla': area
    }])

    # Encode categorical columns and scale numeric
    cat_cols = ['location', 'property_type', 'city']
    num_cols = ['baths', 'bedrooms', 'Area_in_Marla']

    encoded_cat = ohe.transform(input_df[cat_cols])
    scaled_num = scaler.transform(input_df[num_cols])

    # Concatenate all features
    final_input = np.concatenate([encoded_cat, scaled_num], axis=1)

    # Make prediction
    prediction = int(model.predict(final_input)[0])
    formatted_price = f"{prediction:,.0f}"

    return render_template('index.html', cities=cities, property_types=property_types, prediction=formatted_price)

if __name__ == '__main__':
    app.run(debug=True)
