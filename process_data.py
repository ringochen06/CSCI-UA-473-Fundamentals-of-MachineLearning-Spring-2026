import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch

# Configuration
DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_DIR, "tmdb_embedded.parquet")
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def process_data():
    print("🚀 Starting data processing for TMDB 5000 Movies...")
    
    # 1. Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 2. Download Dataset
    print("📥 Downloading dataset from Hugging Face...")
    try:
        # Using a reliable mirror or the original source if possible.
        # The README mentioned AiresPucrs/tmdb-5000-movies
        dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
        df = dataset.to_pandas()
        print(f"✅ Loaded {len(df)} movies.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # 3. Preprocess Text
    print("📝 Preprocessing text...")
    # We need a rich text representation for the model to understand the movie
    # Format: "Title: <title> Overview: <overview> Genres: <genres>"
    
    # Fill NaN values
    df["overview"] = df["overview"].fillna("")
    df["title"] = df["title"].fillna("Unknown Title")
    
    # Genres are often JSON strings or lists, but in this HF dataset they might be simplified.
    # Let's inspect a bit (simulated). If it's the standard dataset, 'genres' is a list of dicts.
    # For simplicity in this script, we'll try to just use overview and title first as they are most important.
    
    # Construct the text to embed
    # Note: Nomic model expects "search_document: " prefix for documents
    df["text_to_embed"] = "search_document: " + \
                          "Title: " + df["title"] + \
                          "; Overview: " + df["overview"]
    
    # 4. Load Model
    print(f"🧠 Loading model {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    
    # 5. Generate Embeddings
    print("⚡ Generating embeddings (this may take a few minutes)...")
    batch_size = 32
    embeddings = model.encode(
        df["text_to_embed"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # 6. Save data
    print("💾 Saving to parquet...")
    df["embedding"] = list(embeddings)
    
    # Also add a 'local_poster_path' column (empty for now) to match app expectations
    if "local_poster_path" not in df.columns:
        df["local_poster_path"] = None
        
    df.to_parquet(OUTPUT_FILE)
    print(f"✅ Data saved to {OUTPUT_FILE}")
    print("🎉 Done!")

if __name__ == "__main__":
    process_data()
