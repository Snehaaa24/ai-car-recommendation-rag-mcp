from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- PREPARE DATA & MODELS ONCE ---
df = pd.read_csv("listings.csv")
rag_model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. RAG Setup: Create search descriptions and embeddings
df['description'] = df['make'] + " " + df['model'] + " " + df['city'] + " " + df['condition']
embeddings = rag_model.encode(df['description'].tolist())

# 2. Price Estimator Setup: Use separate encoders for each column
le_make, le_model, le_city, le_cond = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

df['make_enc'] = le_make.fit_transform(df['make'])
df['model_enc'] = le_model.fit_transform(df['model'])
df['city_enc'] = le_city.fit_transform(df['city'])
df['condition_enc'] = le_cond.fit_transform(df['condition'])

# Define Features and Target
X = df[['make_enc', 'model_enc', 'city_enc', 'condition_enc']]
y = df['price_lakhs']

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('user_query')

    # --- 1. RAG Search Logic ---
    query_vec = rag_model.encode([query])
    # Compare query math to our car listings math
    sims = cosine_similarity(query_vec, embeddings)[0]
    # Get indices of top 3 matches
    top_idx = sims.argsort()[-3:][::-1] 
    
    results = []
    for idx in top_idx:
        # Convert the row to a dictionary so the website can read it
        car = df.iloc[idx].to_dict()
        
        # --- 2. Price Valuation Logic ---
        # Get the predicted price from the column we pre-calculated (or predict live)
        pred_price = rf_model.predict([[car['make_enc'], car['model_enc'], car['city_enc'], car['condition_enc']]])[0]
        car['predicted_price'] = round(pred_price, 2)
        
        # Calculate the % difference (Price Gap)
        diff_percent = (car['price_lakhs'] - pred_price) / pred_price
        
        # Assign status based on your 5% rule
        if diff_percent > 0.05:
            car['valuation'] = "Overpriced"
        elif diff_percent < -0.05:
            car['valuation'] = "Underpriced"
        else:
            car['valuation'] = "Fairly Priced"
            
        results.append(car)
        
    return render_template('index.html', cars=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)