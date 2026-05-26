import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("listings.csv")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create better descriptions for semantic search
df['description'] = (
    df['year'].astype(str) + " " +
    df['make'] + " " +
    df['model'] + " " +
    df['fuel'] + " " +
    df['transmission'] + " " +
    df['kms_driven'].astype(str) + " kms " +
    df['city'] + " " +
    df['condition'] + " condition " +
    df['price_lakhs'].astype(str) + " lakhs"
)

# Create embeddings
embeddings = model.encode(df['description'].tolist())

# Function to search cars
# Function to search cars
def search_cars(query):

    filtered_df = df.copy()

    query_lower = query.lower()

    # Cheap/Budget filter
    if "cheap" in query_lower or "budget" in query_lower:
        filtered_df = filtered_df[
            filtered_df['price_lakhs'] <= 8
        ]

    # SUV filter
    suv_keywords = [
        "suv",
        "family",
        "big car"
    ]

    suv_models = [
        "Creta",
        "Seltos",
        "Nexon",
        "Harrier",
        "Fortuner",
        "Venue",
        "Sonet",
        "EcoSport",
        "WR-V",
        "Vitara Brezza",
        "Hector",
        "Tucson"
    ]

    if any(word in query_lower for word in suv_keywords):

        filtered_df = filtered_df[
            filtered_df['model'].isin(suv_models)
        ]

    # City filter
    cities = [
        "mumbai",
        "pune",
        "delhi",
        "bangalore",
        "chennai"
    ]

    for city in cities:

        if city in query_lower:

            filtered_df = filtered_df[
                filtered_df['city'].str.lower() == city
            ]

    # Create embeddings only for filtered data
    filtered_embeddings = model.encode(
        filtered_df['description'].tolist()
    )

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Compare similarities
    similarities = cosine_similarity(
        query_embedding,
        filtered_embeddings
    )[0]

    # Get top 3 matches
    top_idx = similarities.argsort()[-3:][::-1]

    results = []

    for idx in top_idx:
        results.append(filtered_df.iloc[idx].to_dict())

    return results