# app.py
import time
from src import config
from src.transcribe import transcribe_audio
from src.intent import get_intent
from src.respond import retrieve_knowledge, generate_response
from src.feedback import log_interaction

def run_chatbot_pipeline(audio_path: str, force_language: str = None):
    """
    Runs the full pipeline from audio transcription to response generation
    and logging.
    
    Args:
        audio_path (str): The file path to the user's audio.
        force_language (str, optional): Manually set the audio language.
    """
    
    print(f"\n--- [1/5] Starting Pipeline for: {audio_path} ---")
    
    # --- Task 2: Transcription ---
    transcription_result = transcribe_audio(audio_path, language=force_language)
    if "error" in transcription_result:
        print(f"Pipeline Failed at [Transcription]: {transcription_result['error']}")
        return
    
    text = transcription_result["text"]
    language = transcription_result["language"]
    print(f"   > Language: {language}")
    print(f"   > Text: {text}")

    # --- Task 3: Intent Recognition ---
    print("\n--- [2/5] Recognizing Intent ---")
    intent_result = get_intent(text)
    if "error" in intent_result:
        print(f"Pipeline Failed at [Intent]: {intent_result['error']}")
        return
    
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    print(f"   > Intent: {intent} (Confidence: {confidence:.2f})")

    # --- Task 4: Response Generation (RAG) ---
    print("\n--- [3/5] Retrieving Knowledge ---")
    # We use the full text for retrieval, as it's more accurate
    retrieved_context = retrieve_knowledge(text)
    print(f"   > Retrieved Context: {retrieved_context[:60]}...") # Print snippet

    print("\n--- [4/5] Generating Response ---")
    final_response = generate_response(text, retrieved_context, language)
    
    print("\n" + "="*30)
    print(f"   FINAL RESPONSE ({language}): {final_response}")
    print("="*30 + "\n")

    # --- Task 5: Feedback Loop ---
    print("--- [5/5] Simulating Feedback & Logging ---")
    # In a real app, you'd ask the user. Here, we'll simulate it.
    # Let's say the user was happy if the intent confidence was high.
    user_feedback = "good" if confidence > 0.7 else "bad"
    
    log_interaction(
        language=language,
        text=text,
        intent=intent,
        response=final_response,
        rating=user_feedback
    )
    
    print("--- Pipeline Complete ---")

# --- This block runs when you execute `python app.py` ---
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Put your test audio file here (e.g., from Common Voice)
    TEST_AUDIO_FILE = config.AUDIO_SAMPLES_DIR / "test_es_01.wav"
    
    # You can force a language, or leave as None to auto-detect
    FORCE_LANGUAGE = "spanish" 
    
    # --- EXECUTION ---
    if not TEST_AUDIO_FILE.exists():
        print(f"Error: Test file not found at {TEST_AUDIO_FILE}")
        print("Please download a sample from Common Voice and save it there.")
    else:
        start_time = time.time()
        run_chatbot_pipeline(str(TEST_AUDIO_FILE), force_language=FORCE_LANGUAGE)
        end_time = time.time()
        print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")