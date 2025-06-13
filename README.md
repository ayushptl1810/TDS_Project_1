# TDS Project Scraper

This project scrapes and processes content from the Tools in Data Science (TDS) course forum and website, creating a searchable vector database for efficient information retrieval.

## Prerequisites

- Python 3.12
- Required Python packages (install using `pip install -r requirements.txt`):

## Setup and Execution

Follow these steps in order to set up and run the project:

### 1. Scrape Course Content and Forum Posts

First, run both scraper scripts to collect the data:

```bash
# Scrape the course website content
python scrapper_website.py

# Scrape the forum posts
python scrapper_forum.py
```

This will create:

- `tds_pages_md/` directory containing markdown files of course content
- `tds_forum_posts_filtered.json` containing forum posts
- `tds_forum_topics_filtered.json` containing forum topics

### 2. Process the Scraped Data

Run the processing script to create a simplified, processed version of the documents:

```bash
python processing.py
```

This will create:

- `processed_docs.json` containing cleaned and processed documents

### 3. Create Vector Database

Generate vector embeddings and create the FAISS index:

```bash
python create_vectors.py
```

This will create:

- `vector_store/` directory containing:
  - `md_index.faiss` (vector index for course content)
  - `json_index.faiss` (vector index for forum posts)
  - `md_metadata.pkl` and `json_metadata.pkl` (metadata for the indices)

### 4. Start the Search API Server

Finally, start the API server:

```bash
python api.py
```

The server will start on `http://localhost:8000`. You can then access the api endpoint is at `http://localhost:8000/api/`

## Important Notes

1. Make sure you have the required environment variables set:

   - `GOOGLE_API_KEY` for Gemini AI integration
   - `_t` for Discourse Cookie

2. The scraper scripts use authentication cookies for the forum. Make sure to update the `COOKIES` dictionary in `scrapper_forum.py` with valid credentials.

3. The vector creation process might take some time depending on the amount of content.

4. The API server uses FAISS for efficient similarity search and Gemini AI for generating answers.

## Project Structure

```
.
├── scrapper_website.py    # Scrapes course website content
├── scrapper_forum.py      # Scrapes forum posts and topics
├── processing.py          # Processes and cleans the scraped data
├── create_vectors.py      # Creates vector embeddings and FAISS indices
├── api.py                 # FastAPI server for search and Q&A
├── tds_pages_md/         # Directory for course content markdown files
└── vector_store/         # Directory for FAISS indices and metadata
```
