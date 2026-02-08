from google.cloud import firestore
from config import PROJECT_ID

def get_firestore_client():
    return firestore.Client(project=PROJECT_ID)
