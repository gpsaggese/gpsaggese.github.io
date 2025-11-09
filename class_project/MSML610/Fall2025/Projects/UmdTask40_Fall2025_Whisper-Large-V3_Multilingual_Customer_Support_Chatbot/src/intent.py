# src/intent.py
from transformers import pipeline
from . import config  # Import variables from our config file

# --- 1. Initialize the Intent Classifier Pipeline ---
# This model (facebook/bart-large-mnli) is trained to see
# how well a "premise" (our text) supports a "hypothesis" (our intent labels).
try:
    print(f"Loading Intent model '{config.INTENT_MODEL}' onto {config.DEVICE}...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model=config.INTENT_MODEL,
        device=config.DEVICE
    )
    print("Intent model loaded successfully.")

except Exception as e:
    print(f"Error loading Intent model: {e}")
    classifier = None

# --- 2. Create the Main Function ---

def get_intent(text: str) -> dict:
    """
    Classifies the intent of a given text using zero-shot classification.

    Args:
        text (str): The transcribed text from the user.

    Returns:
        dict: A dictionary containing 'intent' and 'confidence'.
    """
    if classifier is None:
        return {"error": "Intent classifier model is not loaded."}
    
    if not text:
        return {"intent": "unknown", "confidence": 0.0}

    print(f"Classifying intent for text: '{text}'")

    try:
        # Pass the text and our list of candidate labels
        result = classifier(text, config.INTENT_LABELS)
        
        # The result is sorted, so the first label is the best match
        top_intent = result["labels"][0]
        top_score = result["scores"][0]

        print(f"Detected intent: {top_intent} (Confidence: {top_score:.4f})")
        
        return {
            "intent": top_intent,
            "confidence": top_score
        }

    except Exception as e:
        print(f"Error during intent classification: {e}")
        return {"error": f"Intent classification failed: {e}"}

# --- 3. Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    # This block runs ONLY when you execute `python src/intent.py`
    
    # Test 1: A query in English
    test_text_en = "Hello, I can't log in to my account."
    print("\n--- Testing English Intent ---")
    intent_en = get_intent(test_text_en)
    print(f"Result (EN): {intent_en}")
    
    # Test 2: A query in another language (e.g., Spanish)
    # The zero-shot model is multilingual, so it should still work!
    test_text_es = "Hola, no puedo entrar a mi cuenta."
    print("\n--- Testing Spanish Intent ---")
    intent_es = get_intent(test_text_es)
    print(f"Result (ES): {intent_es}") # Should also be 'reset password'

    # Test 3: A different intent
    test_text_billing = "How much do I owe this month?"
    print("\n--- Testing Billing Intent ---")
    intent_billing = get_intent(test_text_billing)
    print(f"Result (Billing): {intent_billing}") # Should be 'billing inquiry'