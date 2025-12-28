import json
import re
import nltk
import spacy
import tarfile
import os
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------
# 1. Configuration and Environment Setup
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data"
CORD19_FILENAME = 'cord-19_2022-06-02.tar.gz'
CORD19_FILE_PATH = str(DATA_DIR / CORD19_FILENAME)
CORD19_URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz"
OUTPUT_DIR = PROJECT_ROOT / "output"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_dataset(url, dest_path, chunk_size=1024*1024):
    """
    Download large files with support for resumable downloads and progress bar.
    """
    dest_path = Path(dest_path)
    
    # 1. Get remote file size
    try:
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to URL: {e}")
        return

    # 2. Check if local file exists and its size
    downloaded_size = 0
    if dest_path.exists():
        downloaded_size = dest_path.stat().st_size
        if downloaded_size == total_size:
            print(f"Dataset already exists and is complete: {dest_path}")
            return
        elif downloaded_size > total_size:
            print("Local file is larger than remote file. Redownloading...")
            downloaded_size = 0
        else:
            print(f"Resuming download from {downloaded_size / (1024**3):.2f} GB...")

    # 3. Set headers for resumable download
    headers = {}
    if downloaded_size > 0:
        headers['Range'] = f'bytes={downloaded_size}-'

    # 4. Start download
    print(f"Downloading dataset to {dest_path}...")
    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            # Progress bar setup
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            with tqdm(total=total_size, initial=downloaded_size, unit='B', unit_scale=True, desc=CORD19_FILENAME, bar_format=bar_format) as pbar:
                with open(dest_path, 'ab') as f: # Use 'ab' (append binary) mode
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print("\nDownload complete!")
    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed: {e}")
        print("Please run the script again to resume.")
        exit(1) # Exit directly on download failure

def setup_nltk():
    """Download necessary NLTK data packages"""
    # Update list, add 'maxent_ne_chunker_tab'
    required_packages = [
        'punkt', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker', 
        'maxent_ne_chunker_tab',
        'words', 
        'punkt_tab'
    ]
    
    print("Checking NLTK packages...")
    for package in required_packages:
        # Call download directly, it automatically skips installed packages, more robust than manual path checking
        nltk.download(package, quiet=True)

setup_nltk()

# ---------------------------------------------------------
# 2. Streaming Data Acquisition
# ---------------------------------------------------------
def stream_cord19_data(tar_path, limit=5):
    """
    Handle nested tar.gz structure:
    Outer tar -> 2022-06-02/document_parses.tar.gz -> pdf_json/*.json
    
    Args:
        tar_path (str): Path to the .tar.gz file
        limit (int): Default to process only a few papers for demonstration. Set to None to process all.
    
    Yields:
        dict: Processed paper data
    """
    if not os.path.exists(tar_path):
        print(f"Error: File not found at {tar_path}")
        return

    print(f"Opening dataset: {tar_path}...")
    
    count = 0
    try:
        # 1. Open outer tar
        with tarfile.open(tar_path, mode="r:gz") as outer_tar:
            # Look for inner document_parses.tar.gz
            inner_tar_member = None
            for member in outer_tar:
                if 'document_parses.tar.gz' in member.name:
                    inner_tar_member = member
                    break
            
            if not inner_tar_member:
                return

            # 2. Extract inner tar as file object (BytesIO)
            f_obj = outer_tar.extractfile(inner_tar_member)
            if f_obj:
                # 3. Open inner tar
                with tarfile.open(fileobj=f_obj, mode="r:gz") as inner_tar:
                    for member in inner_tar:
                        # 4. Look for JSON files
                        if member.isfile() and member.name.endswith('.json'):
                            json_file = inner_tar.extractfile(member)
                            if json_file:
                                try:
                                    content = json.load(json_file)
                                    
                                    # Extract data
                                    paper_id = content.get('paper_id')
                                    title = content.get('metadata', {}).get('title', 'Unknown Title')
                                    body_text_list = content.get('body_text', [])
                                    full_text = " ".join([p.get('text', '') for p in body_text_list])
                                    
                                    if full_text:
                                        yield {
                                            'id': paper_id,
                                            'title': title,
                                            'text': full_text
                                        }
                                        count += 1
                                    
                                    if limit and count >= limit:
                                        return
                                        
                                except Exception:
                                    continue
    except Exception as e:
        print(f"Error: {e}")

# ---------------------------------------------------------
# 3. Text Cleaning
# ---------------------------------------------------------
def clean_text(text):
    """
    Clean text: remove citation markers [1], remove extra whitespace, etc.
    """
    # Remove citations like [1], [12]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Replace special characters but keep punctuation (for sentence splitting)
    text = text.replace('\n', ' ')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 4. Entity Recognition
# ---------------------------------------------------------
def extract_entities_nltk(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(pos_tags)
    
    entities = set() # Use set to remove duplicates
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_name = ' '.join(c[0] for c in chunk)
            # Convert to lowercase for comparison
            entities.add(entity_name.lower())
    return list(entities)

def extract_entities_spacy(text, nlp_model):
    # Truncate overly long text to prevent memory overflow (spaCy default limit 1,000,000 chars)
    if len(text) > 900000:
        text = text[:900000]
    doc = nlp_model(text)
    # Return only entity text, converted to lowercase
    return list(set([ent.text.lower() for ent in doc.ents]))

def extract_entities_transformers(text, pipe):
    # Transformers are slow with long text and have length limits, here we only take the first 512 chars for demo
    # In production, text splitting (sliding window) is needed
    sample_text = text[:512] 
    results = pipe(sample_text)
    entities = set()
    current_entity = ""
    
    for res in results:
        word = res['word']
        if word.startswith("##"):
            current_entity += word[2:]
        else:
            if current_entity:
                entities.add(current_entity.lower())
            current_entity = word
    if current_entity:
        entities.add(current_entity.lower())
    return list(entities)

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
def calculate_metrics(reference_entities, candidate_entities):
    """
    Calculate Precision, Recall, F1
    reference_entities: List of entities treated as "ground truth" (e.g., spaCy results)
    candidate_entities: List of entities to evaluate (e.g., NLTK results)
    """
    ref_set = set(reference_entities)
    cand_set = set(candidate_entities)
    
    # True Positives: Entities found by both models
    tp = len(ref_set.intersection(cand_set))
    # False Positives: Found by candidate model but not by reference model
    fp = len(cand_set - ref_set)
    # False Negatives: Found by reference model but not by candidate model
    fn = len(ref_set - cand_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "overlap_count": tp,
        "nltk_only_count": fp,
        "spacy_only_count": fn
    }

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
def main():
    # 0. Automatically download dataset
    download_dataset(CORD19_URL, CORD19_FILE_PATH)

    # Initialize models
    print("Initializing models...")
    
    # Load spaCy
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return

    # List to store all results
    all_entities_data = []
    performance_metrics = []

    # Process first 10 papers
    LIMIT = 10
    print(f"\n--- Processing {LIMIT} papers from CORD-19 ---")
    
    paper_generator = stream_cord19_data(CORD19_FILE_PATH, limit=LIMIT)
    
    for i, paper in enumerate(paper_generator):
        paper_id = paper['id']
        print(f"Processing [{i+1}/{LIMIT}]: {paper_id}")
        
        cleaned_text = clean_text(paper['text'])
        
        # 1. Run models
        # Note: For fair comparison, we use full text for NLTK too (though slower), or truncate both
        # Here for F1 demo, we use the first 2000 characters
        eval_text = cleaned_text[:2000]
        
        ents_nltk = extract_entities_nltk(eval_text)
        ents_spacy = extract_entities_spacy(eval_text, nlp_spacy)
        
        # 2. Store entity data (for later analysis)
        for ent in ents_nltk:
            all_entities_data.append({'paper_id': paper_id, 'model': 'NLTK', 'entity': ent})
        for ent in ents_spacy:
            all_entities_data.append({'paper_id': paper_id, 'model': 'spaCy', 'entity': ent})
            
        # 3. Calculate performance (using spaCy as Silver Standard)
        metrics = calculate_metrics(reference_entities=ents_spacy, candidate_entities=ents_nltk)
        metrics['paper_id'] = paper_id
        performance_metrics.append(metrics)

    # ---------------------------------------------------------
    # 7. Save and Display Results
    # ---------------------------------------------------------
    
    # Save entity list
    df_entities = pd.DataFrame(all_entities_data)
    entities_csv_path = OUTPUT_DIR / "extracted_entities.csv"
    df_entities.to_csv(entities_csv_path, index=False)
    print(f"\n[Saved] All extracted entities saved to: {entities_csv_path}")
    
    # Save performance metrics
    df_perf = pd.DataFrame(performance_metrics)
    perf_csv_path = OUTPUT_DIR / "performance_metrics.csv"
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[Saved] Performance metrics saved to: {perf_csv_path}")
    
    # Print average performance
    if not df_perf.empty:
        print("\n" + "="*40)
        print("Average Performance (NLTK vs spaCy as Baseline)")
        print("="*40)
        print(df_perf[['precision', 'recall', 'f1_score']].mean())
        print("="*40)
        print("Note: Since CORD-19 is unlabeled, we treat spaCy results as the")
        print("'Silver Standard' (Ground Truth) to evaluate NLTK's relative performance.")

if __name__ == "__main__":
    main()