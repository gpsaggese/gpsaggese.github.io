# validate_data.py
"""
Data validation script for MovieLens dataset.

Validates that downloaded MovieLens data files have the correct structure
and expected columns before running the recommendation system.

Usage:
    python -c "from validate_data import validate_movielens_data; validate_movielens_data('data/raw')"
    # Or as a script:
    python validate_data.py data/raw
"""

import pandas as pd
import os
import sys

def validate_movielens_data(data_dir, check_genome=False):
    """
    Validate that downloaded MovieLens data matches expected structure.
    
    Args:
        data_dir: Path to directory containing rating.csv and movie.csv
        check_genome: If True, also validate genome_tags.csv and genome_scores.csv (optional)
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If columns don't match expected structure
    """
    rating_path = os.path.join(data_dir, "rating.csv")
    movie_path = os.path.join(data_dir, "movie.csv")
    
    # Check files exist
    if not os.path.exists(rating_path):
        raise FileNotFoundError(f"Missing {rating_path}")
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"Missing {movie_path}")
    
    # Check columns
    try:
        ratings = pd.read_csv(rating_path, nrows=1)
        movies = pd.read_csv(movie_path, nrows=1)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")
    
    expected_rating_cols = {'userId', 'movieId', 'rating', 'timestamp'}
    expected_movie_cols = {'movieId', 'title', 'genres'}
    
    actual_rating_cols = set(ratings.columns)
    actual_movie_cols = set(movies.columns)
    
    if actual_rating_cols != expected_rating_cols:
        raise ValueError(
            f"Rating columns mismatch. Expected: {expected_rating_cols}, "
            f"Got: {actual_rating_cols}"
        )
    
    if actual_movie_cols != expected_movie_cols:
        raise ValueError(
            f"Movie columns mismatch. Expected: {expected_movie_cols}, "
            f"Got: {actual_movie_cols}"
        )

    if check_genome:
        genome_tags_path = os.path.join(data_dir, "genome_tags.csv")
        genome_scores_path = os.path.join(data_dir, "genome_scores.csv")
    
    if os.path.exists(genome_tags_path):
        tags = pd.read_csv(genome_tags_path, nrows=1)
        expected_tag_cols = {'tagId', 'tag'}
        if not expected_tag_cols.issubset(tags.columns):
            raise ValueError(f"genome_tags.csv missing required columns")
    
    if os.path.exists(genome_scores_path):
        scores = pd.read_csv(genome_scores_path, nrows=1)
        expected_score_cols = {'movieId', 'tagId', 'relevance'}
        if not expected_score_cols.issubset(scores.columns):
            raise ValueError(f"genome_scores.csv missing required columns")

    print(f"Data validation passed for {data_dir}")
    return True
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <data_directory>")
        print("Example: python validate_data.py data/raw")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    try:
        validate_movielens_data(data_dir)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)