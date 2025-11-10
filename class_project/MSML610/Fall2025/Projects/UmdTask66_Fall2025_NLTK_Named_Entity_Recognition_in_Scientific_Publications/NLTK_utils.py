import nltk
import re
import spacy
import sys
from datasets import load_from_disk

# --- Configuration ---
LOCAL_DATASET_PATH = "/home/haochen/documents/610/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask66_Fall2025_NLTK_Named_Entity_Recognition_in_Scientific_Publications/data"
SPACY_MODEL = "en_core_web_sm"

# --- Helper Functions ---

def download_nltk_data():
    """
    Downloads the necessary NLTK models for tokenization,
    POS tagging, and NER.
    """
    try:
        print("Downloading NLTK components (if not already present)...")
        nltk.download('punkt', quiet=True)        # For tokenization
        nltk.download('averaged_perceptron_tagger', quiet=True) # For POS tagging
        nltk.download('maxent_ne_chunker', quiet=True) # For NER
        nltk.download('words', quiet=True)
        print("NLTK components are ready.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        sys.exit(1)

def load_local_dataset(path):
    """
    Loads the dataset from the local disk path.
    """
    try:
        print(f"Loading dataset from local path: {path}...")
        # load_from_disk loads a DatasetDict
        dataset_dict = load_from_disk(path)
        
        # We'll use the 'train' split for this project
        if 'train' in dataset_dict:
            print("Successfully loaded 'train' split.")
            return dataset_dict['train']
        else:
            print(f"Error: 'train' split not found in local dataset at {path}.")
            return None
    except FileNotFoundError:
        print(f"Error: Local dataset not found at {path}.")
        print("Please run `python download_dataset.py` first.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the local dataset: {e}")
        return None

def clean_text(text):
    """
    Basic text cleaning for a scientific paper abstract.
    - Removes common citation brackets (e.g., [1], [2, 3])
    - Removes URLs
    - Normalizes whitespace
    """
    if not text:
        return ""
    
    text = re.sub(r'\[\d+(, \d+)*\]', '', text) # Remove citations [1], [2, 3]
    text = re.sub(r'\(http[s]?://\S+\)', '', text) # Remove URLs in parentheses
    text = re.sub(r'http[s]?://\S+', '', text)    # Remove standalone URLs
    text = re.sub(r'\s+', ' ', text).strip()   # Normalize whitespace
    return text

def process_with_nltk(text):
    """
    Performs Tokenization, Tagging and 
    Entity Recognition using NLTK.
    """
    print("\n--- NLTK Processing ---")
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Part-of-Speech (POS) Tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Entity Recognition (Chunking)
    # This returns a tree structure
    ner_tree = nltk.ne_chunk(pos_tags)
    
    # Extract entities from the tree for clearer display
    nltk_entities = []
    for chunk in ner_tree:
        # If the chunk is a "tree" (i.e., a named entity)
        if hasattr(chunk, 'label'):
            entity_name = ' '.join(c[0] for c in chunk)
            entity_label = chunk.label()
            nltk_entities.append((entity_name, entity_label))
            
    if not nltk_entities:
        print("NLTK found no named entities.")
    else:
        print(f"Found {len(nltk_entities)} entities (NLTK):")
        for ent in nltk_entities[:10]: # Print first 10
            print(f"  - {ent[0]} ({ent[1]})")

def process_with_spacy(text, nlp_model):
    """
    Performs NER using spaCy for comparison (as suggested in project tasks).
    """
    print("\n--- SpaCy Processing (for comparison) ---")
    doc = nlp_model(text)
    
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    if not spacy_entities:
        print("SpaCy found no named entities.")
    else:
        print(f"Found {len(spacy_entities)} entities (SpaCy):")
        for ent in spacy_entities[:10]: # Print first 10
            print(f"  - {ent[0]} ({ent[1]})")

# --- Main Execution ---

def main():
    # Download NLTK data
    download_nltk_data()

    # Load SpaCy model
    try:
        print(f"Loading SpaCy model '{SPACY_MODEL}'...")
        nlp = spacy.load(SPACY_MODEL)
        print("SpaCy model loaded.")
    except IOError:
        print(f"Error: SpaCy model '{SPACY_MODEL}' not found.")
        print(f"Please run: python -m spacy download {SPACY_MODEL}")
        sys.exit(1)

    # Data Acquisition (from local disk)
    dataset = load_local_dataset(LOCAL_DATASET_PATH)
    if dataset is None:
        sys.exit(1) # Error message already printed in load function
        
    print(f"\nTotal papers in dataset: {len(dataset)}")
    
    # Process a sample paper for feasibility check
    # We use one non-empty abstract as a demonstration
    sample_paper = None
    for paper in dataset:
        if paper['abstract'] and paper['abstract'].strip():
            sample_paper = paper
            break
            
    if sample_paper is None:
        print("Could not find a paper with a valid abstract to process.")
        sys.exit(1)

    print("\n" + "="*50)
    print("Processing a sample paper for NER...")
    print(f"Title: {sample_paper['title']}")
    print("="*50)

    # Text Cleaning
    original_abstract = sample_paper['abstract']
    print(f"\n--- Original Abstract (first 300 chars) ---")
    print(original_abstract[:300] + "...")
    
    cleaned_abstract = clean_text(original_abstract)
    print(f"\n--- Cleaned Abstract (first 300 chars) ---")
    print(cleaned_abstract[:300] + "...")

    # NLTK Tokenization, Tagging, and NER
    process_with_nltk(cleaned_abstract)
    
    # Comparison: spaCy NER
    process_with_spacy(cleaned_abstract, nlp)
    print("\n" + "="*50)
    print("Check processing complete.")
    print("="*50)


if __name__ == "__main__":
    main()