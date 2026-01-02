"""
Main pipeline orchestration script.
Runs all tasks in correct notebook order and logs results + saves plots.
"""

import os
import sys
import pandas as pd

from src import data_loader, preprocess, feature_eng, causal_analysis
from src.utils import setup_logging, log_print, Tee


# ---------------------------------------------------------------------
# Project Root & Directories
# ---------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Results directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(RESULTS_DIR, "output_results.txt")

# Initialize logging system BEFORE starting pipeline
setup_logging(RESULTS_DIR, LOG_FILE)



# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main():

    # Capture all print() output into log file also
    original_stdout = sys.stdout
    log_stream = open(LOG_FILE, "a")
    sys.stdout = Tee(sys.stdout, log_stream)

    try:

        print("\n==============================================================")
        print("Task 1 – Data Acquisition & Weekly Panel Construction")
        print("==============================================================")

        # Load raw dataset
        raw_df = data_loader.download_owid_data()
        print(f" Raw data loaded: {raw_df.shape[0]:,} rows")

        # Minimal cleaning
        base_df = preprocess.clean_data_minimal(raw_df)
        print(f" After minimal cleaning: {base_df.shape[0]:,} country-days")

        # Weekly panel
        weekly_df = preprocess.build_weekly_panel(base_df)
        print(f" Weekly panel created: {weekly_df.shape[0]:,} country-weeks")

        # Features
        panel_df = feature_eng.add_features(weekly_df)
        print(f" Features engineered: {panel_df.shape[0]:,} observations")

        # Final cleaning
        cleaned_df = preprocess.final_clean(panel_df)
        print(f" Final cleaned dataset: {cleaned_df.shape[0]:,} valid country-weeks\n")

        # Save cleaned dataset
        cleaned_df.to_pickle(os.path.join(RESULTS_DIR, "weekly_cleaned.pkl"))
        print("Task 1 completed successfully")
        print("Saved cleaned dataset for analysis\n")

        print("==============================================================")
        print("Task 2–4 + Validation + Policy + Bonus Healthcare Analysis")
        print("==============================================================")
        print("Analysis Executing...\n")

        # Full notebook pipeline
        causal_analysis.run_analysis_workflow(cleaned_df, log_print, PLOTS_DIR)

        print("\n==============================================================")
        print("FINAL OUTPUT COMPLETE")
        print(f"All results saved to: {LOG_FILE}")
        print(f"Plots saved in: {PLOTS_DIR}")
        print("==============================================================\n")

    finally:
        # Restore terminal output
        sys.stdout = original_stdout
        log_stream.close()


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
