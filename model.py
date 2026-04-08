import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv("listings.csv")
print(df.head())

#2. Encode Categorical Data
#Machines only understand numbers, so "Maruti" becomes "1", "Hyundai" becomes "0", etc.
le_make = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_transp = LabelEncoder()

df['make_enc'] = le_make.fit_transform(df['make'])
df['model_enc'] = le_model.fit_transform(df['model'])
df['fuel_enc'] = le_fuel.fit_transform(df['fuel'])
df['trans_enc'] = le_transp.fit_transform(df['transmission'])

# 3. Define Features (X) and Target (y)
X = df[['make_enc', 'model_enc', 'year', 'fuel_enc', 'trans_enc', 'kms_driven']]
y = df['price_lakhs']

# 4. Train the Model
# We'll use a Random Forest (a collection of decision trees)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Predict and Compare
df['predicted_price'] = model.predict(X)

# Logic: Calculate the % difference
# (Listed Price - Predicted Price) / Predicted Price
df['diff_percent'] = (df['price_lakhs'] - df['predicted_price']) / df['predicted_price']

def get_valuation(diff):
    if diff > 0.05:
        return "Overpriced"
    elif diff < -0.05:
        return "Underpriced"
    else:
        return "Fairly Priced"

df['valuation'] = df['diff_percent'].apply(get_valuation)

print(df[['make', 'model', 'price_lakhs', 'predicted_price', 'valuation']])