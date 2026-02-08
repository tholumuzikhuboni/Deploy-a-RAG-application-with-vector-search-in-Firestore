# data/ Directory

This folder is used to store **raw source documents** that will be ingested into the vector database during **Tasks 1 & 2** of the lab.

In this challenge lab, the primary source of truth is a **Food Safety Manual PDF** provided by Google.

---

## What Goes in This Folder?

You should place **raw, unprocessed documents** here, such as:

- PDF manuals (e.g., food safety guides)
- Text files (.txt)
- Markdown files (.md)

For this lab, the main document is:

```
nyc_food_safety_manual.pdf
```

This file is downloaded directly from a Google Cloud Storage bucket during the ingestion step.

---

## How This Folder Is Used in the Lab

### Step 1: Download the PDF

In **Colab Enterprise or Jupyter**, the lab instructs you to download the PDF like this:

```python
!gcloud storage cp gs://partner-genai-bucket/genai069/nyc_food_safety_manual.pdf ./data/
```

This command copies the PDF into the `data/` directory.

---

### Step 2: Load the Document

The PDF is then loaded using a document loader:

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/nyc_food_safety_manual.pdf")
documents = loader.load()
```

At this stage:
- The document is **raw text**
- No embeddings exist yet
- Nothing has been stored in Firestore

---

### Step 3: Clean, Chunk, and Embed

After loading:

1. Text is cleaned (remove line breaks, extra spaces)
2. Text is split into smaller chunks
3. Each chunk is converted into a **vector embedding**
4. Chunks + embeddings are stored in Firestore

Once this step is complete, the `data/` folder is no longer used at runtime.

---

## Important Concept (Beginner Friendly)

Think of this folder like a **library intake desk**:

- Raw books go in (`data/`)
- Knowledge is extracted (embeddings)
- Knowledge is stored elsewhere (Firestore)
- The chatbot never reads the raw book again

---

## Best Practices

- Do **not** commit large PDFs to GitHub unless required
- This folder is mainly for **local ingestion**, not production runtime
- You can add `.gitignore` rules if storing large or temporary files

Example `.gitignore` entry:

```
data/*.pdf
```

---

## Summary

| Folder | Purpose |
|------|--------|
| `data/` | Holds raw documents before ingestion |
| Firestore | Stores processed chunks + embeddings |
| Cloud Run | Serves the chatbot (does NOT read this folder) |

---

You are now ready to move on to **embedding, vector search, and RAG generation**
