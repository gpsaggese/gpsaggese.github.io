# MSML610 Project – TutorTask03_Sentiment_Analysis

##  Goal
Perform sentiment analysis on product reviews using TF-IDF and Logistic Regression.
This project serves as a **hands-on tutorial** to learn how to build, evaluate, and deploy a text-classification model.

---

##  Folder Structure
MSML610/
└── Term2025/
└── projects/
└── TutorTask03_Sentiment_Analysis/

---

## 🧱 Files Overview
| File | Purpose |
|------|----------|
| `sentiment_utils.py` | Helper functions |
| `sentiment.API.ipynb` | API layer demo |
| `sentiment.API.md` | API documentation |
| `sentiment.example.ipynb` | Full example |
| `sentiment.example.md` | Tutorial explanation |
| `Dockerfile` | Container setup |
| `README.md` | Overview & usage guide |

---

## 🐳 Run in Docker
```bash
docker build -t sentiment_tutorial .
docker run -it -p 8888:8888 sentiment_tutorial
