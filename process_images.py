import os
import requests
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
POSTERS_DIR = os.path.join(DATA_DIR, "posters")
INPUT_FILE = os.path.join(PROCESSED_DIR, "tmdb_embedded.parquet")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "tmdb_embedded_with_images.parquet")

MODEL_NAME = "facebook/dinov2-small"

def process_images():
    print("🚀 Starting image processing for TMDB Movies...")
    
    # 1. Load existing data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run process_data.py first.")
        return
        
    print(f"📂 Loading existing data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"✅ Loaded {len(df)} movies.")
    
    # 2. Load Helper Dataset for Poster Paths
    print("📥 Loading helper dataset for poster paths (HenryWaltson/TMDB-IMDB-Movies-Dataset)...")
    try:
        ds_helper = load_dataset("HenryWaltson/TMDB-IMDB-Movies-Dataset", split="train")
        # We only need ID and Poster Path
        # The column names seemed to include 'id' and 'poster_path' based on my check
        df_helper = ds_helper.select_columns(["id", "poster_path"]).to_pandas()
        
        # Ensure ID is same type for merge
        df["id"] = df["id"].astype(str)
        df_helper["id"] = df_helper["id"].astype(str)
        
        # Drop duplicates in helper
        df_helper = df_helper.drop_duplicates(subset=["id"])
        
        print(f"✅ Loaded {len(df_helper)} helper records.")
        
        # Merge
        print("🔄 Merging to find poster paths...")
        # Left join to keep all our original movies
        # Since original df does NOT have 'poster_path', the merged df will simply acquire 'poster_path' from helper
        # If original df DOES have 'poster_path' (e.g. from previous run), we use suffixes
        
        df = df.merge(df_helper, on="id", how="left", suffixes=("", "_helper"))
        
        # Logic to coalesce poster_path
        if "poster_path_helper" in df.columns:
             # This means both had 'poster_path'. We prefer the one that is not null, or helper if both exist?
             # Actually, original probably has None if it exists.
             if "poster_path" in df.columns:
                 df["poster_path"] = df["poster_path"].fillna(df["poster_path_helper"])
             else:
                 df["poster_path"] = df["poster_path_helper"]
             del df["poster_path_helper"]
        
        # If no conflict, 'poster_path' is already there from helper (or original)
        if "poster_path" not in df.columns:
            print("⚠️ Warning: poster_path column missing after merge")
            df["poster_path"] = None

            
        found_count = df["poster_path"].notna().sum()
        print(f"✅ Found poster paths for {found_count}/{len(df)} movies.")
        
    except Exception as e:
        print(f"❌ Failed to load helper dataset: {e}")
        return

    # 3. Download Images
    print("⬇️  Downloading posters...")
    os.makedirs(POSTERS_DIR, exist_ok=True)
    
    local_paths = []
    
    # Create the column if it doesn't exist
    if "local_poster_path" not in df.columns:
        df["local_poster_path"] = None

    # We use a session for connection pooling
    session = requests.Session()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        movie_id = row["id"]
        poster_path = row["poster_path"]
        
        # Define local filename
        filename = f"{movie_id}.jpg"
        filepath = os.path.join(POSTERS_DIR, filename)
        
        # Check if already exists
        if os.path.exists(filepath):
            local_paths.append(filepath)
            continue
            
        # Check if we have a URL to download
        if pd.isna(poster_path):
            local_paths.append(None)
            continue
            
        try:
            # Construct URL
            url = f"https://image.tmdb.org/t/p/w200{poster_path}"
            
            response = session.get(url, timeout=5)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                local_paths.append(filepath)
            else:
                local_paths.append(None)
        except Exception as e:
            # print(f"Error downloading {movie_id}: {e}")
            local_paths.append(None)
            
    df["local_poster_path"] = local_paths
    
    # 4. Generate Image Embeddings
    print(f"🧠 Loading DINOv2 model ({MODEL_NAME})...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    print("⚡ Generating image embeddings...")
    
    image_embeddings = []
    
    # Process one by one for simplicity and robustness (batching images with different sizes/failures is tricky)
    # Since we have only 5000, it's manageable.
    
    for filepath in tqdm(df["local_poster_path"], desc="Embedding"):
        if filepath and os.path.exists(filepath):
            try:
                image = Image.open(filepath).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # DINOv2 CLS token
                    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    # Normalize
                    emb = emb / np.linalg.norm(emb)
                    image_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                image_embeddings.append(None)
        else:
            image_embeddings.append(None)
            
    df["image_embedding"] = image_embeddings
    
    # 5. Save
    print("💾 Saving updated dataset...")
    # Drop rows without image embeddings? Optional. We'll keep them but they have None.
    # Note: Parquet can format None arrays tricky-ly.
    # We will fill None with NaN or just zeros if needed, but let's try to save as is first.
    
    # Parquet doesn't like mixed types in object columns sometimes.
    # Let's clean up the image_embedding column.
    
    # Replace None with empty list or NaN? 
    # Best is to leave as None and ensure column is object.
    
    df.to_parquet(OUTPUT_FILE)
    
    # Also overwrite the input file so the app picks it up
    df.to_parquet(INPUT_FILE)
    
    print(f"✅ Data saved to {INPUT_FILE}")
    print("🎉 Done!")

if __name__ == "__main__":
    process_images()
