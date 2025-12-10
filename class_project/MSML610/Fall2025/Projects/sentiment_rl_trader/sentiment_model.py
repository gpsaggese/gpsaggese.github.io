# sentiment_model.py
# This file handles the sentiment analysis part of the project.
# We use the VADER model to score how positive or negative each news title is.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# create the sentiment analyzer once
analyzer = SentimentIntensityAnalyzer()

def add_sentiment(df):
    # some news titles might be empty, so we replace None with ""
    df["sentiment"] = df["title"].fillna("").apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    return df
