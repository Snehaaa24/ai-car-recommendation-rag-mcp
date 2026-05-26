import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("listings.csv")

# Create encoders
le_make = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

# Encode categorical columns
df['make_enc'] = le_make.fit_transform(df['make'])
df['model_enc'] = le_model.fit_transform(df['model'])
df['fuel_enc'] = le_fuel.fit_transform(df['fuel'])
df['trans_enc'] = le_trans.fit_transform(df['transmission'])

# Features and target
X = df[['make_enc', 'model_enc', 'year', 'fuel_enc', 'trans_enc', 'kms_driven']]
y = df['price_lakhs']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to check price
def check_price(make, model_name, year, fuel, transmission, kms_driven, listed_price):

    # Encode incoming values
    make_enc = le_make.transform([make])[0]
    model_enc = le_model.transform([model_name])[0]
    fuel_enc = le_fuel.transform([fuel])[0]
    trans_enc = le_trans.transform([transmission])[0]

    # Create feature row
    features = [[
        make_enc,
        model_enc,
        year,
        fuel_enc,
        trans_enc,
        kms_driven
    ]]

    # Predict price
    predicted_price = model.predict(features)[0]

    # Compare prices
    diff_percent = (listed_price - predicted_price) / predicted_price

    if diff_percent > 0.05:
        verdict = "Overpriced"
    elif diff_percent < -0.05:
        verdict = "Underpriced"
    else:
        verdict = "Fairly Priced"

    return {
        "estimated_price": round(predicted_price, 2),
        "listed_price": listed_price,
        "verdict": verdict
    }