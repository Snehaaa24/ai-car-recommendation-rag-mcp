#rag research logic
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load your data
df = pd.read_csv("listings.csv")

# 2. Initialize the model 
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Create the "Search Index" (Turn all car rows into math)
# We combine columns so the AI has more context to read
df['description'] = df['make'] + " " + df['model'] + " " + df['city'] + " " + df['price_lakhs'].astype(str) + " Lakhs " + df['condition']
embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)

#define the query
query="cheap first car for a student"

#turn it into math for the embeddings
query_vector=model.encode([query])

#Compare the query to all your car listings (using cosine similarity)
similarities=cosine_similarity(query_vector,embeddings)

# find the best match
best_idx=similarities.argmax()
print(f"\nTop match for '{query}'")
print(df.iloc[best_idx])