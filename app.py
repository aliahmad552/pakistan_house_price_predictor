from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# === Load Models and Preprocessors ===
working_dir = os.path.dirname(os.path.abspath(__file__))

# Models
sale_model = pickle.load(open(f'{working_dir}/model/xgb.pkl', 'rb'))
rent_model = pickle.load(open(f'{working_dir}/model/xgb_forRent.pkl', 'rb'))

# Encoder & Scaler
ohe = pickle.load(open(f'{working_dir}/model/ohe.pkl', 'rb'))
scaler = pickle.load(open(f'{working_dir}/model/scaler.pkl', 'rb'))

# Data for dropdowns
df = pd.read_csv(f'{working_dir}/model/Cleaned_data.csv')
cities = sorted(df['city'].unique())
property_types = sorted(df['property_type'].unique())


# === Routes ===

@app.route('/')
def index():
    """Render the home page with dropdown options."""
    return render_template('index.html', cities=cities, property_types=property_types)

@app.template_filter('comma')
def comma_filter(value):
    try:
        return "{:,}".format(int(value))
    except (ValueError, TypeError):
        return value

@app.route('/get_locations/<city>')
def get_locations(city):
    """Return list of locations for a selected city (AJAX request)."""
    filtered = df[df['city'] == city]
    locations = sorted(filtered['location'].unique())
    return jsonify({'locations': locations})


@app.route('/', methods=['POST'])
def predict():
    """Handle prediction requests and return results with extra insights."""
    # --- Collect Form Data ---
    city = request.form['city']
    location = request.form['location']
    property_type = request.form['property_type']
    bedrooms = int(request.form['bedrooms'])
    baths = int(request.form['baths'])
    area = float(request.form['area'])
    model_type = request.form['model_type']  # 'sale' or 'rent'

    # --- Select Model ---
    model = sale_model if model_type == 'sale' else rent_model

    # --- Prepare Input Data ---
    input_df = pd.DataFrame([{
        'city': city,
        'location': location,
        'property_type': property_type,
        'bedrooms': bedrooms,
        'baths': baths,
        'Area_in_Marla': area
    }])

    cat_cols = ['location', 'property_type', 'city']
    num_cols = ['baths', 'bedrooms', 'Area_in_Marla']

    encoded_cat = ohe.transform(input_df[cat_cols])
    scaled_num = scaler.transform(input_df[num_cols])
    final_input = np.concatenate([encoded_cat, scaled_num], axis=1)

    # --- Make Prediction ---
    prediction = int(model.predict(final_input)[0])
    formatted_price = f"{prediction:,.0f}"

    # --- Additional Insights ---
    # Filter dataset to the same location & city
    location_data = df[(df['city'] == city) & (df['location'] == location)]
    if not location_data.empty:
        avg_price = int(location_data['price'].mean())
        min_price = int(location_data['price'].min())
        max_price = int(location_data['price'].max())
        total_listings = len(location_data)
    else:
        avg_price = min_price = max_price = total_listings = None

    # Prepare small list of price samples for visualization (histogram)
    price_distribution = location_data['price'].sample(min(30, len(location_data))).tolist() if not location_data.empty else []

    # --- Render Page with All Info ---

    location_data = df[df['city'] == city].groupby('location')['price'].mean().sort_values(ascending=False).head(10)
    price_distribution = location_data.values.tolist()
    location_labels = location_data.index.tolist()

    return render_template(
        'index.html',
        cities=cities,
        property_types=property_types,
        prediction=formatted_price,
        city=city,
        location=location,
        property_type=property_type,
        bedrooms=bedrooms,
        baths=baths,
        area=area,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price,
        total_listings=total_listings,
        price_distribution=price_distribution,
        model_type=model_type,
        location_labels=location_labels
    )


# === Main ===
if __name__ == '__main__':
    app.run(debug=True)
