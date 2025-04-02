from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Sample dataset
data = {
    'size': [850, 900, 1200, 1500, 1800, 2100, 2500],
    'location': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago', 'New York'],
    'price': [300000, 280000, 350000, 270000, 320000, 290000, 400000]
}

df = pd.DataFrame(data)
df['location'] = df['location'].str.lower()  # Convert to lowercase

# Encode locations using One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
location_encoded = encoder.fit_transform(df[['location']])
location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['location']))

# Merge data
df_encoded = pd.concat([df[['size']], location_df, df['price']], axis=1)

# Train model
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    size = float(request.form['size'])
    location = request.form['location'].strip().lower()

    # Check if location is valid
    if location not in df['location'].unique():
        return "Error: Invalid location! Please enter New York, Los Angeles, or Chicago."

    # Encode user location
    user_location_df = pd.DataFrame(encoder.transform(pd.DataFrame([[location]], columns=['location'])), 
                                    columns=encoder.get_feature_names_out(['location']))

    # Create DataFrame for prediction
    user_input_df = pd.DataFrame([[size]], columns=['size'])
    user_input_final = pd.concat([user_input_df, user_location_df], axis=1)

    # Predict price
    predicted_price = model.predict(user_input_final)[0]

    # Render the result page
    return render_template('result.html', size=size, location=location.title(), price=f"${predicted_price:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
