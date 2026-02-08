from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from langchain_google_vertexai import VertexAIEmbeddings
from config import PROJECT_ID, EMBEDDING_MODEL_NAME, COLLECTION_NAME
from utils.firestore_client import get_firestore_client

db = get_firestore_client()
collection = db.collection(COLLECTION_NAME)

embedding_model = VertexAIEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, 
    project=PROJECT_ID
)

def search_vector_database(query: str):
    # Convert text query to vector
    query_embedding = embedding_model.embed_query(query)

    # Perform Nearest Neighbor search
    results = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=5
    ).get()

    context = "\n".join([doc.to_dict().get("content", "") for doc in results])
    return context
