# Semantic Search Engine

**Author:** Ali Fehmi Yildiz  
**UID:** 121326737  
**Project:** MSML610 Final Project  
**Difficulty:** Medium

---

## What I Built

I built a search engine that understands what you mean, not just the words you type.

**Example:**
- I search: "famous tower in Paris"
- Regular search: No results (those exact words not found)
- My search: Finds the Eiffel Tower! 

My system understands that "famous tower in Paris" means the Eiffel Tower, even though I never typed those exact words.

---

## Why This Matters

Regular search engines only find exact word matches. If you don't know the right words, you don't find what you need.

My search engine understands meaning:
- I can search for "red planet" and find Mars
- I can search for "AI and neural networks" and find machine learning articles
- I can ask questions in my own words and still get good results

This is called **semantic search** - searching by meaning instead of matching words.

---

## What I Used

**Data:** I used 50,000 Wikipedia articles  
**AI Model:** Sentence Transformers (all-MiniLM-L6-v2)  
**Programming:** Python with Flask for the website  
**Tools:** Jupyter notebooks, Docker

---

## How to Use It

### Option 1: Docker (Easiest)
```bash
docker-compose up
```
Then open your browser to: http://localhost:5000

### Option 2: Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open your browser to: http://localhost:5000



---

## How It Works

I broke the problem into simple steps:

**Step 1: Load the Data**
- I downloaded 50,000 Wikipedia articles
- Each article is a piece of text about a topic

**Step 2: Convert Text to Numbers**
- I used an AI model to turn each article into 384 numbers
- These numbers capture the "meaning" of the article
- Articles about similar topics get similar numbers

**Step 3: Search**
- When you type a question, I convert it to 384 numbers too
- I compare your question's numbers with all the articles' numbers
- I find which articles have the most similar numbers
- Similar numbers = similar meaning = good match!

**Step 4: Show Results**
- I show you the best matches
- The articles that match your meaning, not just your words

---

## The Math Part (Simple Explanation)

I use something called **cosine similarity**. Think of it like measuring angles:

- If two vectors (articles) point in the same direction → They're about similar topics
- If they point in different directions → They're about different topics

The closer the direction, the better the match!

---

## What's in This Project

```
My Project Files:
├── semantic_search_complete.ipynb  ← My full tutorial (look here first!)
├── app.py                          ← The web application
├── requirements.txt                ← Python packages I used
├── Dockerfile                      ← Docker setup
├── docker-compose.yml              ← Easy Docker start
├── data_sample.parquet             ← Sample Wikipedia data
└── README.md                       ← This file
```

---

## My Results

I tested my search with different queries:

**Test 1:** "famous tower in Paris"
- Result: Found Eiffel Tower articles 
- Score: 0.54 similarity

**Test 2:** "artificial intelligence"
- Result: Found AI and machine learning articles 
- Multiple relevant matches

**Test 3:** "red planet"
- Result: Found Mars and planetary articles 
- Even though "Mars" wasn't in my search

**What I learned:**
- Semantic search works! It understands meaning.
- Most of my 50,000 articles have low similarity to any single query (this is good - it means I'm finding the specific relevant ones)
- Search is fast - about 20 milliseconds per query

---

## Problems I Solved

**Problem 1:** The AI model was too big to download
- Solution: I used a smaller, faster model (all-MiniLM-L6-v2)

**Problem 2:** Processing 50,000 articles took too long
- Solution: I process them once, then save the results

**Problem 3:** Docker containers were too large
- Solution: I used a small sample dataset for Docker (5,000 articles)

**Problem 4:** Keeping track of texts and embeddings
- Solution: I made sure to always reload both together

---

## If You Want to Try It

1. **Make sure you have Python 3.9+**

2. **Clone this project:**
   ```bash
   git clone <repo-url>
   cd semantic-search
   ```

3. **Install what you need:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run it:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   ```
   http://localhost:5000
   ```

6. **Search for anything!**
   Try: "space exploration" or "famous scientists"

---

## What I Learned

**Technical Skills:**
- How to use Sentence Transformers for AI
- How to build vector search systems
- How to deploy with Docker
- How to create web applications with Flask

**Important Concepts:**
- Embeddings (turning words into numbers that capture meaning)
- Cosine similarity (measuring how similar meanings are)
- Semantic search (searching by meaning, not words)

**Real-World Applications:**
- This technology powers modern search engines
- It's used in chatbots and question-answering systems
- Companies use it for document search and recommendations

---

## References

I learned from these sources:

1. **Sentence Transformers Documentation**  
   https://www.sbert.net/

2. **The Research Paper**  
   "Sentence-BERT" by Reimers & Gurevych (2019)

3. **Wikipedia Dataset**  
   https://www.kaggle.com/datasets/jjinho/wikipedia-20230701

4. **Flask Documentation**  
   https://flask.palletsprojects.com/

---

## Contact

**Ali Fehmi Yildiz**  
University of Maryland  
UID: 121326737

---

*I built this project for MSML610 - December 2025*