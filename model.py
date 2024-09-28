import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn
import logging
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='book_recommendation.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data (adjust the path if necessary)
data = pd.read_csv('data/books.csv')

def get_all_book_titles():
    return data['Title'].tolist()  # Return all titles as a list

# Handle missing values
data['Author'] = data['Author'].fillna('Unknown Author')
data['Publisher'] = data['Publisher'].fillna('Unknown Publisher')

# Combine relevant features into a single string for each book
data['combined_features'] = data['Title'] + " " + data['Author'] + " " + data['Publisher'] + " " + data['Genre'] + " " + data['SubGenre']

# Vectorize the combined features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Apply K-Means clustering and log the model using MLflow
num_clusters = 5  # Adjust the number of clusters as needed

with mlflow.start_run():
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(tfidf_matrix)
    inertia = kmeans.inertia_
    
    logging.info(f"KMeans model trained with inertia: {inertia}")

    # Create a valid input example based on the TF-IDF features
    sample_input = tfidf.transform(["Example Book Title Example Author Example Publisher Example Genre Example SubGenre"])
    sample_input = sample_input.toarray()

    # Define input and output example for the model signature
    input_example = pd.DataFrame(sample_input)

    # Define the model signature
    signature = ModelSignature(inputs=Schema([
        ColSpec(type="float", name="tfidf_features")  # The input type for KMeans should be float
    ]), outputs=Schema([
        ColSpec(type="integer", name="cluster")  # Adjust according to your model's output
    ]))

    # Log the model with the signature and input example
    mlflow.sklearn.log_model(kmeans, "kmeans_model", signature=signature, input_example=input_example)
    logging.info("KMeans model logged successfully in MLflow with signature and input example.")

# Function to get book recommendations based on the same cluster
def get_recommendations(title):
    # Check if 'cluster' column exists
    if 'cluster' not in data.columns:
        raise KeyError("The 'cluster' column was not created. Check the KMeans process.")
    
    try:
        # Get the cluster of the requested book
        book_cluster = data[data['Title'].str.lower() == title.lower()]['cluster'].values[0]
    except IndexError:
        raise IndexError("Book not found in the dataset.")
    
    # Get books in the same cluster
    similar_books = data[data['cluster'] == book_cluster]['Title'].tolist()
    
    # Remove the requested book from the list of recommendations
    if title in similar_books:
        similar_books.remove(title)

    # Log the recommendation request
    logging.info(f"Recommendations for '{title}' fetched successfully.")
    
    # Return the top 5 recommendations from the same cluster
    return similar_books[:5]
