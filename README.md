# FreshBot: A RAG App with Firestore & Gemini 2.0

![FreshBot Banner](https://github.com/tholumuzikhuboni/Deploy-a-RAG-application-with-vector-search-in-Firestore/blob/ee86a686fee6792ebcf3f443c2c4ccc36f1b71e1/HHLBQBk5%2BMXjSDVwBEFj57dIt61d7ms2VdkwS9CFJNw%3D.png)

**Lab Code:** `GENAI069`  
**Target Model:** `Gemini 2.0 Flash`  
**Database:** Firestore (Native Mode)  
**Deployment:** Cloud Run  

FreshBot is a **Retrieval-Augmented Generation (RAG)** application that answers food safety questions. It combines:

- **Firestore**: stores text chunks and embeddings  
- **Vertex AI embeddings**: converts text into vector representations  
- **Gemini 2.0 Flash**: generates answers based on retrieved context  
- **Cloud Run**: hosts the Flask application  

This repo is designed to take a beginner **step-by-step**, from PDF ingestion to deploying a fully functional web app.

---

##  Prerequisites

Before starting, make sure you have:

- A **Google Cloud project** with billing enabled  
- **Cloud Shell** or local terminal with `gcloud` CLI installed  
- Access to **Firestore Native Mode**  
- Optional: Familiarity with Python & Flask  

---

##  Repository Structure

```
rag-food-safety-bot/
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerize Flask app for Cloud Run
├── main.py                    # Flask application
├── config.py                  # Configuration file (project ID, region)
├── utils/                     # Helper modules
│   ├── firestore_client.py    # Firestore client wrapper
│   ├── vector_utils.py        # Embedding & vector search
│   └── genai_utils.py         # Gemini query functions
├── templates/                 # HTML templates
│   └── index.html
├── notebooks/                 # Jupyter notebook for ingestion
│   └── ingest_data.ipynb
└── data/                      # PDFs / raw text
    └── README.md
```

> This separation of concerns makes it easy to maintain, debug, and extend.

---

##  Phase 1: Load & Chunk PDF (Tasks 1 & 2)

FreshBot uses a **food safety manual PDF** as its knowledge base.  
We first load it, clean the text, and split it into chunks for embedding.

### Install dependencies:

```bash
python3 -m pip install --upgrade \
    google-cloud-logging google-cloud-firestore \
    google-cloud-aiplatform langchain langchain-experimental==0.3.4 \
    langchain-community langchain-google-vertexai pymupdf
```

### Notebook: `ingest_data.ipynb`

- Load PDF with **PyMuPDFLoader**  
- Clean text (remove newlines, whitespace)  
- Chunk text with **SemanticChunker**  
- Embed chunks with **VertexAIEmbeddings**  
- Upload text + embeddings to Firestore  

```python
from google.cloud import firestore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from google.cloud.firestore_v1.vector import Vector

# Firestore client
db = firestore.Client(project="YOUR_PROJECT_ID")
collection = db.collection("food-safety")

# Load PDF
loader = PyMuPDFLoader("nyc_food_safety_manual.pdf")
pages = loader.load()

# Clean pages
cleaned_pages = [p.page_content.replace("\n"," ") for p in pages]

# Chunk pages
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")
text_splitter = SemanticChunker(embedding_model)
chunks = text_splitter.create_documents(cleaned_pages[:5])

# Generate embeddings & store
for doc in chunks:
    embedding = embedding_model.embed_documents([doc.page_content])[0]
    collection.add({"content": doc.page_content, "embedding": Vector(embedding)})
```

>  Now your Firestore database contains vectorized chunks for semantic search.

---

## Phase 2: Firestore Vector Index (Task 3)

To search vectors efficiently, create a **vector index**:

```bash
gcloud firestore indexes composite create \
    --project=YOUR_PROJECT_ID \
    --collection-group=food-safety \
    --query-scope=COLLECTION \
    --field-config=vector-config='{"dimension":"768","flat":"{}"}',field-path=embedding
```

> Firestore uses this index to return **nearest neighbor chunks** efficiently.

---

## Phase 3: Flask App Deployment (Task 4)

FreshBot uses **Flask** to serve the web app.  

### `main.py` Overview:

- Connects to Firestore collection  
- Initializes **VertexAI embeddings** and **Gemini generative model**  
- Provides **`search_vector_database()`** to retrieve context  
- Provides **`ask_gemini()`** to generate answer  
- Flask route renders HTML template  

**Key safety setting:** `BLOCK_ONLY_HIGH` prevents Gemini from suggesting harmful actions (e.g., knives, burns).

### Run Locally (Cloud Shell)

```bash
python3 main.py
```

Preview app:

- Click **Web Preview → Preview on port 8080**  
- Ask: `What temperature range do Mesophilic Bacteria grow best in?`  
- Expected output: `50 to 110 degrees Fahrenheit`

---

## Docker & Cloud Run

### Step 1: Set Artifact Registry

```bash
export ARTIFACT_REPO=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/cymbal-artifact-repo
```

### Step 2: Build & Push Docker Image

```bash
docker build -t $ARTIFACT_REPO/cymbal-docker-image -f Dockerfile .
docker push $ARTIFACT_REPO/cymbal-docker-image
```

### Step 3: Deploy to Cloud Run

```bash
gcloud run deploy cymbal-freshbot \
    --image=$ARTIFACT_REPO/cymbal-docker-image \
    --allow-unauthenticated \
    --region=us-central1
```

Visit the URL and test the app!

---

## How FreshBot Works (RAG Explained)

1. **User asks a question** → Flask receives input  
2. **Query embedding** → Convert question into a vector using VertexAI  
3. **Vector search** → Firestore returns top 5 most similar chunks  
4. **Generative answer** → Gemini 2.0 Flash answers using only the retrieved context  
5. **Display answer** → Flask renders it in the browser  

> This is **Retrieval-Augmented Generation (RAG)** in action.

---

## Tips for Beginners

- Firestore Native Mode = document-based NoSQL database  
- Vector embeddings = numeric representations of text  
- Gemini = Large language model for generative answers  
- Cloud Run = fully managed container hosting, auto-scaling  

---

## Final Test

Ask:

> "What temperature range do Mesophilic Bacteria grow best in?"  

Expected response:

> `50 to 110 degrees Fahrenheit`

---

## References

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)  
- [Firestore Vector Search](https://cloud.google.com/firestore/docs/solutions/vector-search)  
- [Cloud Run Overview](https://cloud.google.com/run/docs)  
- [Gemini 2.0 Flash](https://cloud.google.com/vertex-ai/docs/generative-ai/gemini-overview)

---
```
THOLUMUZI KUBONI
```

