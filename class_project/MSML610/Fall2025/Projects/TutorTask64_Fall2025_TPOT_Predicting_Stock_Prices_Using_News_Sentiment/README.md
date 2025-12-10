Project Directory

TutorTask64_Fall2025_TPOT_Predicting_Stock_Prices_Using_News_Sentiment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── docker/                           # Docker configuration
│   ├── Dockerfile                    # Container definition
│   ├── docker_build.sh              # Build script
│   └── docker_run.sh                # Run script
│
├── notebooks/                        # Jupyter notebooks
│   ├── TPOT.API.ipynb              # TPOT API exploration
│   ├── TPOT.example.ipynb          # Full example application
│   ├── TPOT_utils.py               # Utility functions
│   └── data_processing.py          # Data processing utilities
│
├── data/                            # Data files
│   ├── Data_readme.md              # Data documentation
│   ├── daily_news_sentiment.parquet
│   ├── valid_tickers.csv
│   ├── tpot_best_pipeline.py       # Exported TPOT pipeline
│   └── tpot_fitted_model.pkl       # Trained model
│
└── docs/                           # Documentation (if applicable)
    ├── TPOT.API.md
    └── TPOT.example.md
