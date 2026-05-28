import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("listings.csv")

# -------------------------
# Convert rows into documents
# -------------------------
documents = []

for _, row in df.iterrows():

    content = f"""
    {row['year']} {row['make']} {row['model']}
    Fuel: {row['fuel']}
    Transmission: {row['transmission']}
    Kms Driven: {row['kms_driven']}
    City: {row['city']}
    Condition: {row['condition']}
    Price: {row['price_lakhs']} lakhs
    """

    doc = Document(
        page_content=content,
        metadata=row.to_dict()
    )

    documents.append(doc)

# -------------------------
# Local Embedding Model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Create Chroma Vector DB
# -------------------------
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# -------------------------
# Retriever
# -------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Search Function
# -------------------------
def search_cars(query):

    query_lower = query.lower()

    filtered_df = df.copy()

    active_filters = []

    # -------------------------
    # Budget / Cheap Cars
    # -------------------------
    budget_keywords = [
        "cheap",
        "budget",
        "affordable",
        "economical",
        "student",
        "first car"
    ]

    if any(word in query_lower for word in budget_keywords):

        filtered_df = filtered_df[
            filtered_df['price_lakhs'] <= 8
        ]

        active_filters.append("budget")

    # -------------------------
    # Premium / Luxury Cars
    # -------------------------
    luxury_keywords = [
        "premium",
        "luxury",
        "expensive",
        "high-end"
    ]

    if any(word in query_lower for word in luxury_keywords):

        filtered_df = filtered_df[
            filtered_df['price_lakhs'] >= 12
        ]

        active_filters.append("premium")

    # -------------------------
    # SUV / Family Cars
    # -------------------------
    suv_keywords = [
        "suv",
        "family",
        "spacious",
        "big car",
        "road trip",
        "7 seater"
    ]

    suv_models = [
        "Creta",
        "Seltos",
        "Nexon",
        "Harrier",
        "EcoSport",
        "Venue",
        "Sonet",
        "Tucson",
        "Fortuner",
        "Vitara Brezza",
        "Hector",
        "WR-V",
        "Ertiga",
        "Innova"
    ]

    if any(word in query_lower for word in suv_keywords):

        filtered_df = filtered_df[
            filtered_df['model'].isin(suv_models)
        ]

        active_filters.append("suv")

    # -------------------------
    # Automatic Cars
    # -------------------------
    automatic_keywords = [
        "automatic",
        "auto",
        "traffic"
    ]

    if any(word in query_lower for word in automatic_keywords):

        filtered_df = filtered_df[
            filtered_df['transmission'] == "Automatic"
        ]

        active_filters.append("automatic")

    # -------------------------
    # Diesel Cars
    # -------------------------
    if "diesel" in query_lower:

        filtered_df = filtered_df[
            filtered_df['fuel'] == "Diesel"
        ]

        active_filters.append("diesel")

    # -------------------------
    # Electric Cars
    # -------------------------
    electric_keywords = [
        "electric",
        "ev",
        "battery"
    ]

    if any(word in query_lower for word in electric_keywords):

        filtered_df = filtered_df[
            filtered_df['fuel'] == "Electric"
        ]

        active_filters.append("electric")

    # -------------------------
    # Low KMs / Less Driven
    # -------------------------
    low_km_keywords = [
        "low kms",
        "low mileage",
        "less driven",
        "new car"
    ]

    if any(word in query_lower for word in low_km_keywords):

        filtered_df = filtered_df[
            filtered_df['kms_driven'] <= 30000
        ]

        active_filters.append("low_kms")

    # -------------------------
    # City Filters
    # -------------------------
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

            active_filters.append(city)

    # -------------------------
    # SAFETY FALLBACK
    # -------------------------
    if len(filtered_df) == 0:

        filtered_df = df.copy()

    # -------------------------
    # Create Temporary Documents
    # -------------------------
    temp_documents = []

    for _, row in filtered_df.iterrows():

        content = f"""
        {row['year']} {row['make']} {row['model']}
        Fuel: {row['fuel']}
        Transmission: {row['transmission']}
        Kms Driven: {row['kms_driven']}
        City: {row['city']}
        Condition: {row['condition']}
        Price: {row['price_lakhs']} lakhs
        """

        doc = Document(
            page_content=content,
            metadata=row.to_dict()
        )

        temp_documents.append(doc)

    # -------------------------
    # Temporary Vector Store
    # -------------------------
    temp_vector_store = Chroma.from_documents(
        documents=temp_documents,
        embedding=embedding_model
    )

    temp_retriever = temp_vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    results = temp_retriever.invoke(query)

    # -------------------------
    # Remove Duplicates
    # -------------------------
    cars = []

    seen_ids = set()

    for result in results:

        car_data = result.metadata

        if car_data['id'] not in seen_ids:

            matched_reasons = []
            missing_reasons = []

            # -------------------------
            # SUV Match
            # -------------------------
            if "suv" in active_filters:

                if car_data['model'] in suv_models:
                    matched_reasons.append("SUV")
                else:
                    missing_reasons.append("SUV")

            # -------------------------
            # Automatic Match
            # -------------------------
            if "automatic" in active_filters:

                if car_data['transmission'] == "Automatic":
                    matched_reasons.append("Automatic")
                else:
                    missing_reasons.append("Automatic")

            # -------------------------
            # Budget Match
            # -------------------------
            if "budget" in active_filters:

                if car_data['price_lakhs'] <= 8:
                    matched_reasons.append("Budget Friendly")
                else:
                    missing_reasons.append("Budget")

            # -------------------------
            # Premium Match
            # -------------------------
            if "premium" in active_filters:

                if car_data['price_lakhs'] >= 12:
                    matched_reasons.append("Premium")
                else:
                    missing_reasons.append("Premium")

            # -------------------------
            # Diesel Match
            # -------------------------
            if "diesel" in active_filters:

                if car_data['fuel'] == "Diesel":
                    matched_reasons.append("Diesel")
                else:
                    missing_reasons.append("Diesel")

            # -------------------------
            # Electric Match
            # -------------------------
            if "electric" in active_filters:

                if car_data['fuel'] == "Electric":
                    matched_reasons.append("Electric")
                else:
                    missing_reasons.append("Electric")

            # -------------------------
            # Low KMs Match
            # -------------------------
            if "low_kms" in active_filters:

                if car_data['kms_driven'] <= 30000:
                    matched_reasons.append("Low KMs")
                else:
                    missing_reasons.append("Low KMs")

            # -------------------------
            # City Match
            # -------------------------
            for city in cities:

                if city in active_filters:

                    if car_data['city'].lower() == city:
                        matched_reasons.append(city.title())
                    else:
                        missing_reasons.append(city.title())

            # -------------------------
            # Save Reasons
            # -------------------------
            car_data['matched_reasons'] = matched_reasons
            car_data['missing_reasons'] = missing_reasons

            cars.append(car_data)

            seen_ids.add(car_data['id'])

    return cars