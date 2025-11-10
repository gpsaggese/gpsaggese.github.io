
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import json
from google.cloud import storage
from google.cloud import aiplatform
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# DATA INGESTION FUNCTIONS
def load_twitter_data(file_path: str) -> pd.DataFrame:
    """
    Load the Twitter airline sentiment dataset.

    :param file_path: Path to the CSV file
    :return: DataFrame containing the Twitter data
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully loaded {len(df)} tweets from {file_path}")
        return df    
    except Exception as e:
        raise FileNotFoundError(f"Data file not found: {file_path}")


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about the dataset.

    :param df: DataFrame to analyze
    :return: Dictionary containing dataset statistics
    """
    info = {
        'total_records': len(df),
        'num_columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
    }

    if 'airline_sentiment' in df.columns:
        info['sentiment_distribution'] = df['airline_sentiment'].value_counts().to_dict()
        info['sentiment_percentages'] = (df['airline_sentiment'].value_counts(normalize=True) * 100).to_dict()

    if 'airline' in df.columns:
        info['airline_distribution'] = df['airline'].value_counts().to_dict()

    return info


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.

    :param df: DataFrame to summarize
    """
    info = get_dataset_info(df)

    print("DATASET SUMMARY")
    print(f"\n Basic Information:")
    print(f"Total Records: {info['total_records']:,}")
    print(f"Number of Columns: {info['num_columns']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")

    print(f"\n Columns:")
    for col in info['column_names']:
        dtype = str(info['data_types'][col])
        missing = info['missing_values'][col]
        missing_pct = (missing / info['total_records']) * 100
        print(f"{col:30s} - Type: {dtype:10s} - Missing: {missing:6,} ({missing_pct:5.2f}%)")

    if 'sentiment_distribution' in info:
        print(f"\n Sentiment Distribution:")
        for sentiment, count in info['sentiment_distribution'].items():
            pct = info['sentiment_percentages'][sentiment]
            print(f"{sentiment:10s}: {count:6,} tweets ({pct:5.2f}%)")

    if 'airline_distribution' in info:
        print(f"\n Airline Distribution:")
        for airline, count in sorted(info['airline_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / info['total_records']) * 100
            print(f"{airline:20s}: {count:6,} tweets ({pct:5.2f}%)")


def visualize_sentiment_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Visualize the distribution of sentiments in the dataset.

    :param df: DataFrame containing 'airline_sentiment' column
    :param save_path: Optional path to save the figure
    """
    if 'airline_sentiment' not in df.columns:
        raise ValueError("DataFrame does not contain 'airline_sentiment' column")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sentiment_counts = df['airline_sentiment'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    sentiment_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]

    axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=sentiment_colors, alpha=0.8)
    axes[0].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')

    axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=sentiment_colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Sentiment Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_airline_sentiment(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Visualize sentiment distribution across different airlines.

    :param df: DataFrame containing 'airline' and 'airline_sentiment' columns
    :param save_path: Optional path to save the figure
    """
    if 'airline' not in df.columns or 'airline_sentiment' not in df.columns:
        raise ValueError("DataFrame does not contain 'airline' and 'airline_sentiment' columns")

    airline_sentiment = pd.crosstab(df['airline'], df['airline_sentiment'], normalize='index') * 100

    if 'negative' in airline_sentiment.columns:
        airline_sentiment = airline_sentiment.sort_values('negative', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
    airline_sentiment.plot(kind='bar', stacked=True, ax=axes[0],
                           color=[colors.get(col, '#95a5a6') for col in airline_sentiment.columns],
                           alpha=0.8)
    axes[0].set_xlabel('Airline', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Sentiment Distribution by Airline', fontsize=14, fontweight='bold')
    axes[0].legend(title='Sentiment', title_fontsize=11, fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    airline_counts = pd.crosstab(df['airline'], df['airline_sentiment'])
    airline_counts.plot(kind='bar', ax=axes[1],
                       color=[colors.get(col, '#95a5a6') for col in airline_counts.columns],
                       alpha=0.8)
    axes[1].set_xlabel('Airline', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Tweet Count by Airline and Sentiment', fontsize=14, fontweight='bold')
    axes[1].legend(title='Sentiment', title_fontsize=11, fontsize=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def analyze_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Analyze text statistics like length, word count, etc.

    :param df: DataFrame containing text data
    :param text_column: Name of the column containing text
    :return: DataFrame with text statistics
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{text_column}' column")

    stats_df = df.copy()

    stats_df['text_length'] = stats_df[text_column].astype(str).apply(len)
    stats_df['word_count'] = stats_df[text_column].astype(str).apply(lambda x: len(x.split()))
    stats_df['avg_word_length'] = stats_df.apply(
        lambda row: row['text_length'] / row['word_count'] if row['word_count'] > 0 else 0, axis=1
    )
    stats_df['has_url'] = stats_df[text_column].astype(str).str.contains('http', case=False, na=False)
    stats_df['has_mention'] = stats_df[text_column].astype(str).str.contains('@', na=False)
    stats_df['has_hashtag'] = stats_df[text_column].astype(str).str.contains('#', na=False)
    stats_df['exclamation_count'] = stats_df[text_column].astype(str).str.count('!')
    stats_df['question_count'] = stats_df[text_column].astype(str).str.count(r'\?')

    return stats_df


def print_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> None:
    """
    Print summary statistics about the text data.

    :param df: DataFrame containing text data
    :param text_column: Name of the column containing text
    """
    stats_df = analyze_text_statistics(df, text_column)

    print("TEXT STATISTICS")

    print(f"\n Text Length Statistics:")
    print(f"Mean: {stats_df['text_length'].mean():.2f} characters")
    print(f"Median: {stats_df['text_length'].median():.2f} characters")
    print(f"Min: {stats_df['text_length'].min()} characters")
    print(f"Max: {stats_df['text_length'].max()} characters")
    print(f"Std Dev: {stats_df['text_length'].std():.2f} characters")

    print(f"\n Word Count Statistics:")
    print(f"Mean: {stats_df['word_count'].mean():.2f} words")
    print(f"Median: {stats_df['word_count'].median():.2f} words")
    print(f"Min: {stats_df['word_count'].min()} words")
    print(f"Max: {stats_df['word_count'].max()} words")

    print(f"\n Content Features:")
    print(f"Tweets with URLs: {stats_df['has_url'].sum():,} ({stats_df['has_url'].mean()*100:.2f}%)")
    print(f"Tweets with Mentions: {stats_df['has_mention'].sum():,} ({stats_df['has_mention'].mean()*100:.2f}%)")
    print(f"Tweets with Hashtags: {stats_df['has_hashtag'].sum():,} ({stats_df['has_hashtag'].mean()*100:.2f}%)")
    print(f"Average Exclamation Marks: {stats_df['exclamation_count'].mean():.2f}")
    print(f"Average Question Marks: {stats_df['question_count'].mean():.2f}")

    # Statistics by sentiment
    if 'airline_sentiment' in df.columns:
        print(f"\nText Length by Sentiment:")
        for sentiment in stats_df['airline_sentiment'].unique():
            sentiment_data = stats_df[stats_df['airline_sentiment'] == sentiment]
            mean_length = sentiment_data['text_length'].mean()
            mean_words = sentiment_data['word_count'].mean()
            print(f"{sentiment:10s}: {mean_length:6.2f} chars, {mean_words:5.2f} words")


def sample_tweets_by_sentiment(df: pd.DataFrame, n_samples: int = 3,
                               text_column: str = 'text') -> None:
    """
    Display sample tweets for each sentiment category.

    :param df: DataFrame containing tweets
    :param n_samples: Number of samples per sentiment
    :param text_column: Name of the column containing text
    """
    if 'airline_sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'airline_sentiment' column")

    print("SAMPLE TWEETS BY SENTIMENT")

    sentiments = df['airline_sentiment'].unique()

    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in sentiments:
            print(f"{sentiment.upper()} TWEETS")
            

            samples = df[df['airline_sentiment'] == sentiment].sample(
                n=min(n_samples, len(df[df['airline_sentiment'] == sentiment])),
                random_state=42
            )

            for idx, (_, row) in enumerate(samples.iterrows(), 1):
                tweet_text = row[text_column]
                airline = row.get('airline', 'Unknown')
                confidence = row.get('airline_sentiment_confidence', 'N/A')

                print(f"\n{idx}. Airline: {airline} | Confidence: {confidence}")
                print(f"   Tweet: {tweet_text}")


def prepare_data_for_vertex_ai(df: pd.DataFrame, text_column: str = 'text',
                               label_column: str = 'airline_sentiment',
                               output_path: str = 'data/processed/sentiment_data.jsonl') -> str:
    """
    Prepare data in JSONL format for Vertex AI training.

    :param df: DataFrame containing the data
    :param text_column: Name of the text column
    :param label_column: Name of the label column
    :param output_path: Path to save the JSONL file
    :return: Path to the created JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare data in JSONL format
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_obj = {
                "text_content": str(row[text_column]),
                "category": str(row[label_column])
            }
            f.write(json.dumps(json_obj) + '\n')

    print(f"Saved to: {output_path}")

    return output_path


def split_train_val_test(df: pd.DataFrame, train_ratio: float = 0.7,
                         val_ratio: float = 0.15, test_ratio: float = 0.15,
                         random_state: int = 42, stratify_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.

    :param df: DataFrame to split
    :param train_ratio: Proportion for training set
    :param val_ratio: Proportion for validation set
    :param test_ratio: Proportion for test set
    :param random_state: Random seed for reproducibility
    :param stratify_column: Column name to stratify split (e.g., 'airline_sentiment')
    :return: Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Prepare stratification
    stratify = df[stratify_column] if stratify_column else None

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify
    )

    # Second split: separate train and validation
    # Calculate validation size relative to train_val
    val_size_relative = val_ratio / (train_ratio + val_ratio)

    stratify_train_val = train_val_df[stratify_column] if stratify_column else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=random_state,
        stratify=stratify_train_val
    )

    print(f"Training set:   {len(train_df):6,} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df):6,} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:       {len(test_df):6,} records ({len(test_df)/len(df)*100:.1f}%)")

    if stratify_column:
        print(f"\n  Sentiment distribution verification:")
        for dataset_name, dataset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            dist = dataset[stratify_column].value_counts(normalize=True) * 100
            print(f"   {dataset_name:5s}: ", end="")
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment in dist.index:
                    print(f"{sentiment}: {dist[sentiment]:5.2f}%  ", end="")

    return train_df, val_df, test_df


def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True,
               remove_hashtags: bool = False, remove_numbers: bool = False,
               remove_punctuation: bool = False) -> str:
    """
    Clean text by removing or replacing various elements.

    :param text: Input text to clean
    :param remove_urls: Remove URLs (http/https links)
    :param remove_mentions: Remove @mentions
    :param remove_hashtags: Remove hashtags (keeps the text after #)
    :param remove_numbers: Remove numbers
    :param remove_punctuation: Remove punctuation marks
    :return: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML entities
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)

    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    if remove_hashtags:
        text = re.sub(r'#(\w+)', r'\1', text)

    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.

    :param text: Input text to tokenize
    :return: List of tokens
    """
    try:
        return word_tokenize(text)
    except LookupError:
        print("NLTK punkt tokenizer not found. Using simple split.")
        return text.split()


def remove_stopwords(tokens: List[str], keep_negation: bool = True) -> List[str]:
    """
    Remove stop words from list of tokens.

    :param tokens: List of tokens
    :param keep_negation: Keep negation words like 'not', 'no', 'never' (important for sentiment)
    :return: List of tokens without stop words
    """
    try:
        stop_words = set(stopwords.words('english'))

        # Keep negation words for sentiment analysis
        if keep_negation:
            negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                            'nowhere', 'none', 'barely', 'hardly', 'scarcely', "n't", 'dont', 'doesn'}
            stop_words = stop_words - negation_words

        return [token for token in tokens if token.lower() not in stop_words]
    except LookupError:
        print("NLTK stopwords not found.")
        return tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens to their base form.

    :param tokens: List of tokens to lemmatize
    :return: List of lemmatized tokens
    """
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        print("NLTK wordnet not found")
        return tokens


def preprocess_tweet(text: str, remove_stopwords_flag: bool = True,
                    lemmatize: bool = True, keep_negation: bool = True) -> str:
    """
    Complete preprocessing pipeline for tweets.

    :param text: Input tweet text
    :param remove_stopwords_flag: Whether to remove stop words
    :param lemmatize: Whether to lemmatize tokens
    :param keep_negation: Keep negation words (important for sentiment)
    :return: Preprocessed text
    """
    cleaned = clean_text(
        text,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_numbers=False,
        remove_punctuation=True
    )

    tokens = tokenize_text(cleaned)

    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens, keep_negation=keep_negation)

    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text',
                        output_column: str = 'text_processed',
                        **preprocessing_kwargs) -> pd.DataFrame:
    """
    Apply preprocessing to all tweets in a DataFrame.

    :param df: DataFrame containing tweets
    :param text_column: Name of column containing raw text
    :param output_column: Name for the preprocessed text column
    :param **preprocessing_kwargs: Additional arguments for preprocess_tweet()
    :return: DataFrame with added preprocessed text column
    """
    print(f"Preprocessing {len(df)} tweets...")

    df = df.copy()
    df[output_column] = df[text_column].apply(
        lambda x: preprocess_tweet(str(x), **preprocessing_kwargs)
    )

    avg_original_length = df[text_column].astype(str).apply(len).mean()
    avg_processed_length = df[output_column].astype(str).apply(len).mean()
    reduction = ((avg_original_length - avg_processed_length) / avg_original_length) * 100

    print(f"Average original length: {avg_original_length:.1f} chars")
    print(f"Average processed length: {avg_processed_length:.1f} chars")
    print(f"Length reduction: {reduction:.1f}%")

    return df


def analyze_preprocessing_impact(df: pd.DataFrame, text_column: str = 'text',
                                 processed_column: str = 'text_processed') -> None:
    """
    Analyze the impact of preprocessing on the text data.

    :param df: DataFrame with both original and processed text
    :param text_column: Name of original text column
    :param processed_column: Name of processed text column
    """

    original_lengths = df[text_column].astype(str).apply(len)
    processed_lengths = df[processed_column].astype(str).apply(len)

    original_words = df[text_column].astype(str).apply(lambda x: len(x.split()))
    processed_words = df[processed_column].astype(str).apply(lambda x: len(x.split()))

    print(f"\n Text Length Statistics:")
    print(f"   Original:")
    print(f"  Mean: {original_lengths.mean():.1f} chars")
    print(f"  Median: {original_lengths.median():.1f} chars")
    print(f"   Processed:")
    print(f"  Mean: {processed_lengths.mean():.1f} chars")
    print(f"  Median: {processed_lengths.median():.1f} chars")
    print(f"   Reduction: {((original_lengths.mean() - processed_lengths.mean()) / original_lengths.mean() * 100):.1f}%")

    print(f"\n Word Count Statistics:")
    print(f"   Original:")
    print(f"  Mean: {original_words.mean():.1f} words")
    print(f"  Median: {original_words.median():.1f} words")
    print(f"   Processed:")
    print(f"  Mean: {processed_words.mean():.1f} words")
    print(f"  Median: {processed_words.median():.1f} words")
    print(f"   Reduction: {((original_words.mean() - processed_words.mean()) / original_words.mean() * 100):.1f}%")

    for i, (_, row) in enumerate(df.sample(5, random_state=42).iterrows(), 1):
        print(f"\nExample {i}:")
        print(f"  Original:  {row[text_column][:100]}...")
        print(f"  Processed: {row[processed_column][:100]}...")


if __name__ == "__main__":
    print("Vertex AI Sentiment Analysis Utilities")
