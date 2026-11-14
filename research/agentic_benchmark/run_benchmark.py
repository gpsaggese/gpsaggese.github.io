import sys
import pandas as pd
import agentic_bench as ab
from langchain_ollama import OllamaLLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("data/amazon_beauty_50k.csv")
df = df.dropna(subset=["text", "title", "rating"])
df = df[(df["text"].str.len() > 0) & (df["title"].str.len() > 0)]
df = df.sample(5000, random_state=42)          # sample first so VADER stays fast

df["sentiment"] = (df["rating"] >= 4).astype(int)   # target
df["text_length"] = df["text"].str.len()
df["title_length"] = df["title"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["exclaim_count"] = df["text"].str.count("!")
df["upper_ratio"] = df["text"].apply(
    lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
)
vader = SentimentIntensityAnalyzer()
df["vader_text"] = df["text"].apply(lambda s: vader.polarity_scores(s)["compound"])
df["vader_title"] = df["title"].apply(lambda s: vader.polarity_scores(s)["compound"])
df["verified"] = df["verified"].astype(int)

# rating is NOT a feature, it would leak the target
cols = ["text_length", "title_length", "word_count", "exclaim_count",
        "upper_ratio", "vader_text", "vader_title", "helpful_vote",
        "verified", "sentiment"]
df = df[cols].dropna()

datasets = {"Amazon_Reviews": (df, "sentiment")}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rules"   # "rules" or "llm"
    llm = OllamaLLM(model="llama3.2") if mode == "llm" else None
    out = f"outputs/results_amazon_{mode}.csv"
    ab.run_experiment(
        datasets,
        seeds=range(5),
        conditions=[(0.0, 0.0), (0.05, 0.02), (0.15, 0.05), (0.30, 0.05)],
        llm=llm,
        strategy_mode=mode,
        out_csv=out,
        verbose=True,
    )
    print(ab.summarize(out))
    print(ab.detector_table(out))