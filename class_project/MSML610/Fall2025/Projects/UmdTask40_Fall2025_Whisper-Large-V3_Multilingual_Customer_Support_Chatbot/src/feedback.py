# src/feedback.py
import csv
from datetime import datetime
from . import config  # Import variables from our config file

# --- 1. Ensure Log Directory and File Exists ---
LOG_FILE = config.FEEDBACK_LOG_PATH
LOG_HEADER = [
    "timestamp", 
    "language", 
    "transcribed_text", 
    "recognized_intent", 
    "response", 
    "rating" # e.g., "good" or "bad"
]

try:
    # Create the 'logs/' directory if it doesn't exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file and write the header if it's new
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)
        print(f"Created new feedback log at: {LOG_FILE}")
        
except Exception as e:
    print(f"Error setting up log file: {e}")

# --- 2. Create the Main Logging Function ---

def log_interaction(language: str, text: str, intent: str, response: str, rating: str):
    """
    Appends a single chatbot interaction to the feedback log.

    Args:
        language (str): Detected language (e.g., 'spanish')
        text (str): The transcribed text from the user
        intent (str): The recognized intent (e.g., 'reset password')
        response (str): The final response given to the user
        rating (str): The user's feedback (e.g., 'good', 'bad', 'thumb_up')
    """
    try:
        # 1. Get the current time
        timestamp = datetime.now().isoformat()
        
        # 2. Create the new row of data
        new_row = [timestamp, language, text, intent, response, rating]
        
        # 3. Append to the CSV file
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
            
        print(f"Successfully logged interaction. Rating: {rating}")

    except Exception as e:
        print(f"Error logging interaction: {e}")


# --- 3. Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    # This block runs ONLY when you execute `python src/feedback.py`
    
    print("\n--- Testing Feedback Logger ---")
    
    # Test 1: A "good" interaction
    log_interaction(
        language="english",
        text="I need to reset my password.",
        intent="reset password",
        response="To reset your password, please go to the login page...",
        rating="good"
    )
    
    # Test 2: A "bad" interaction (e.g., intent was wrong)
    log_interaction(
        language="spanish",
        text="Dónde está mi orden?",
        intent="billing inquiry",  # This was the (wrong) recognized intent
        response="For all billing inquiries, please contact our support...", # This is the (wrong) response
        rating="bad" # User flags this as bad
    )
    
    print(f"\nCheck the log file at: {LOG_FILE}")