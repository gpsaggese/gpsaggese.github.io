# src/respond.py
import json
import torch
from . import config
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# --- 1. Load Knowledge Base ---
def load_knowledge_base():
    """Loads the FAQ data from the JSON file."""
    try:
        with open(config.KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        print(f"Knowledge base loaded successfully from {config.KNOWLEDGE_BASE_PATH}")
        return kb_data
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {config.KNOWLEDGE_BASE_PATH}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config.KNOWLEDGE_BASE_PATH}")
        return []

# --- 2. Initialize Models and Knowledge Index ---
try:
    # Load all KB data (list of dictionaries)
    knowledge_base = load_knowledge_base()
    
    # Extract just the 'answer' texts to be indexed
    knowledge_contexts = [item['answer'] for item in knowledge_base]

    # Load Embedding Model (for RAG retrieval)
    print(f"Loading embedding model '{config.EMBEDDING_MODEL}'...")
    embedding_model = SentenceTransformer(
        config.EMBEDDING_MODEL, 
        device=config.DEVICE
    )

    # Load Generative LLM (for RAG generation)
    print(f"Loading LLM '{config.LLM_MODEL}' for generation...")
    llm = pipeline(
        "text2text-generation",
        model=config.LLM_MODEL,
        device=config.DEVICE,
        torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32
    )

    # --- Create the Knowledge Index ---
    # This runs ONCE when the module is loaded.
    # We convert all our FAQ answers into vector embeddings.
    if knowledge_contexts:
        print("Creating embeddings for knowledge base... (This may take a moment)")
        kb_embeddings = embedding_model.encode(
            knowledge_contexts, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("Knowledge base indexing complete.")
    else:
        print("Warning: Knowledge base is empty. RAG retrieval will not work.")
        kb_embeddings = None

except Exception as e:
    print(f"Error initializing models or knowledge base: {e}")
    embedding_model = None
    llm = None
    kb_embeddings = None
    knowledge_base = []

# --- 3. Retrieval Function ---
def retrieve_knowledge(query_text: str) -> str:
    """
    Finds the most relevant knowledge base entry for a user query.
    
    Args:
        query_text (str): The user's transcribed text.

    Returns:
        str: The most relevant 'answer' from the knowledge base.
    """
    if embedding_model is None or kb_embeddings is None or not knowledge_base:
        return "I am sorry, but my knowledge base is currently unavailable."
    
    # 1. Embed the user's query
    query_embedding = embedding_model.encode(
        query_text, 
        convert_to_tensor=True
    ).to(config.DEVICE)

    # 2. Perform semantic search (Cosine Similarity)
    # This finds the closest match between the query and all KB embeddings
    hits = util.semantic_search(
        query_embedding, 
        kb_embeddings, 
        top_k=1
    )
    
    # We only care about the top hit (top_k=1)
    best_hit = hits[0][0]
    best_hit_index = best_hit['corpus_id']
    best_hit_score = best_hit['score']

    print(f"Retrieval: Best hit index {best_hit_index}, Score: {best_hit_score:.4f}")

    # 3. Return the actual text of the best-matching answer
    return knowledge_base[best_hit_index]['answer']

# --- 4. Generation Function ---
def generate_response(user_query: str, retrieved_context: str, language: str) -> str:
    """
    Generates a final response using the LLM, grounded in the retrieved context.

    Args:
        user_query (str): The user's original, transcribed question.
        retrieved_context (str): The relevant info from our knowledge base.
        language (str): The language the user spoke (e.g., 'spanish', 'french').

    Returns:
        str: The final, translated, and grounded response.
    """
    if llm is None:
        return "I am sorry, but my response generator is not working."
    
    # This prompt is CRITICAL.
    # It forces the LLM to use *only* our context and to reply
    # in the user's language.
    prompt = f"""
    You are a customer support agent.
    Answer the user's question in {language}.
    
    User's Question: "{user_query}"
    
    Use ONLY this information to answer:
    "{retrieved_context}"
    
    If the information is not enough, politely say you cannot help in {language}.
    
    Answer in {language}:
    """

    print("Generating final response...")
    try:
        # Generate the response
        response = llm(prompt, max_length=150, num_beams=4, early_stopping=True)
        final_text = response[0]['generated_text']
        
        print(f"Generated response: {final_text}")
        return final_text

    except Exception as e:
        print(f"Error during LLM response generation: {e}")
        return f"I apologize, I encountered an error trying to respond in {language}."

# --- 5. Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    # This block runs ONLY when you execute `python src/respond.py`
    
    # Test 1: Spanish query for order status
    test_query_es = "Hola, dónde está mi pedido?"
    test_lang_es = "spanish"
    
    print(f"\n--- Testing RAG flow for: '{test_query_es}' ---")
    # Step 1: Retrieve
    context_es = retrieve_knowledge(test_query_es)
    print(f"Retrieved Context: {context_es}")
    # Step 2: Generate
    response_es = generate_response(test_query_es, context_es, test_lang_es)
    print(f"Final Response (ES): {response_es}")


    # Test 2: English query for password
    test_query_en = "I can't log in"
    test_lang_en = "english"
    
    print(f"\n--- Testing RAG flow for: '{test_query_en}' ---")
    # Step 1: Retrieve
    context_en = retrieve_knowledge(test_query_en)
    print(f"Retrieved Context: {context_en}")
    # Step 2: Generate
    response_en = generate_response(test_query_en, context_en, test_lang_en)
    print(f"Final Response (EN): {response_en}")