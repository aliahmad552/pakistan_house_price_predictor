from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained models and data
model = pickle.load(open('xgb.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load cleaned data to get available options
df = pd.read_csv('Cleaned_data.csv')

# Define the exact feature order expected by the model
CATEGORICAL_FEATURES = ['location', 'property_type', 'city', 'purpose']
NUMERICAL_FEATURES = ['bedrooms', 'bathrooms', 'Area_in_Marla']


@app.route('/')
def home():
    # Get unique values for all dropdowns
    cities = sorted(df['city'].unique().tolist())
    purposes = sorted(df['purpose'].unique().tolist())
    property_types = sorted(df['property_type'].unique().tolist())
    return render_template('index.html',
                           cities=cities,
                           purposes=purposes,
                           property_types=property_types)


@app.route('/get_locations', methods=['POST'])
def get_locations():
    city = request.json['city']
    locations = sorted(df[df['city'] == city]['location'].unique().tolist())
    return jsonify({'locations': locations})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Prepare features in EXACTLY the same order as during training
        categorical_features = {
            'location': data['location'],
            'property_type': data['property_type'],
            'city': data['city'],
            'purpose': data['purpose']
        }

        numerical_features = {
            'bedrooms': float(data['bedrooms']),
            'bathrooms': float(data['bathrooms']),
            'Area_in_Marla': float(data['Area_in_Marla'])
        }

        # Create DataFrames maintaining the exact feature order
        cat_df = pd.DataFrame([categorical_features])[CATEGORICAL_FEATURES]
        num_df = pd.DataFrame([numerical_features])[NUMERICAL_FEATURES]

        # One Hot Encoding for categorical features
        cat_encoded = ohe.transform(cat_df).toarray()

        # Scaling for numerical features
        num_scaled = scaler.transform(num_df)

        # Combine features
        final_features = np.concatenate([cat_encoded, num_scaled], axis=1)

        # Make prediction
        prediction = model.predict(final_features)

        # Format the prediction for display
        output = round(prediction[0], 2)
        formatted_output = "{:,.2f} PKR".format(output)

        return render_template('index.html',
                               prediction_text=f'Predicted Price: {formatted_output}',
                               cities=sorted(df['city'].unique().tolist()),
                               purposes=sorted(df['purpose'].unique().tolist()),
                               property_types=sorted(df['property_type'].unique().tolist()),
                               form_data=data)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error in prediction: {str(e)}',
                               cities=sorted(df['city'].unique().tolist()),
                               purposes=sorted(df['purpose'].unique().tolist()),
                               property_types=sorted(df['property_type'].unique().tolist()))


if __name__ == "__main__":
    app.run(debug=True)