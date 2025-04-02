import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Sample dataset: House size, location, and corresponding price
data = {
    'size': [850, 900, 1200, 1500, 1800, 2100, 2500],
    'location': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago', 'New York'],
    'price': [300000, 280000, 350000, 270000, 320000, 290000, 400000]
}

df = pd.DataFrame(data)

# Convert location names to lowercase for consistency
df['location'] = df['location'].str.lower()

# Encode locations into numerical values
encoder = OneHotEncoder(sparse=False)
location_encoded = encoder.fit_transform(df[['location']])
location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['location']))

# Merge encoded locations with the dataset
df_encoded = pd.concat([df[['size']], location_df, df['price']], axis=1)

# Split data into features (X) and target (y)
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Take user input
user_size = float(input("Enter the house size in square feet: "))
user_location = input("Enter the location (New York, Los Angeles, Chicago): ").strip().lower()  # Convert to lowercase

# Check if location is valid
if user_location not in df['location'].unique():
    print("Error: Invalid location! Please enter New York, Los Angeles, or Chicago.")
else:
    # Fix: Convert user location into a DataFrame with feature names
    user_location_df = pd.DataFrame(encoder.transform(pd.DataFrame([[user_location]], columns=['location'])), 
                                    columns=encoder.get_feature_names_out(['location']))

    # Create DataFrame for prediction
    user_input_df = pd.DataFrame([[user_size]], columns=['size'])
    user_input_final = pd.concat([user_input_df, user_location_df], axis=1)

    # Predict price
    predicted_price = model.predict(user_input_final)
    print(f"Estimated Price for a {user_size} sq. ft house in {user_location.title()}: ₹{predicted_price[0]:,.2f}")
