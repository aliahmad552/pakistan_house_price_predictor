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

        # Input
        purpose = data['purpose']
        city = data['city']
        location = data['location']
        property_type = data['property_type']
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        area = float(data['Area_in_Marla'])

        # Create DataFrame
        input_df = pd.DataFrame({
            'city': [city],
            'location': [location],
            'purpose': [purpose],
            'property_type': [property_type],
            'bedrooms': [bedrooms],
            'baths': [bathrooms],
            'Area_in_Marla': [area]
        })

        # Encode categorical
        cat_cols = ['purpose','location','property_type', 'city']
        X_cat = ohe.transform(input_df[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_cols))

        # Scale numerical
        num_cols = ['baths','bedrooms','Area_in_Marla']
        X_num = scaler.transform(input_df[num_cols])
        X_num_df = pd.DataFrame(X_num, columns=num_cols)

        # Combine
        X_final = pd.concat([X_num_df.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)

        # Load feature order
        with open('feature_names.pkl', 'rb') as f:
            expected_features = pickle.load(f)

        # Reorder and fill missing if needed
        X_final = X_final.reindex(columns=expected_features, fill_value=0)

        # Predict
        pred = model.predict(X_final)[0]
        return jsonify({'predicted_price': round(pred)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
