# Description

CLIP-ViT-Large-Patch14 is a vision-language model from OpenAI that embeds images and
text into a shared space, so that an image can be compared with any natural-language
description. It solves image classification and retrieval without task-specific
training (zero-shot) and adapts to a new task with a few labeled images through a
linear probe on the frozen embeddings. It is worth a 60-minute tutorial because a few
lines of code give a working zero-shot classifier and a reusable image-text embedding
API.

## Technologies Used

CLIP-ViT-Large-Patch14

- Combines visual and textual understanding to perform image-text matching
- Supports zero-shot learning, allowing the model to classify images without explicit
  training on specific categories
- Utilizes a transformer architecture for efficient processing of visual and textual
  data
- Exposes image and text embeddings through `CLIPModel` and `CLIPProcessor`, which
  can be reused for linear probes and retrieval

# Tutorial

- Implement the tutorial "Learn CLIP in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 CLIP project, and reuse what is good
    - `class_project/msml610/Fall2025/projects/TutorTask37_Fall2025_CLIP_ViT_Large_Patch14_Generative_Art_from_Text_Prompts/`
  - Look at the code of the second Fall2025 CLIP project, which has no `README.md`
    - `class_project/msml610/Fall2025/projects/CLIP_ViT_Large_Task22/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `clip_utils.py`
  - `clip.API.ipynb`
  - `clip.example.ipynb`

# Project

## Project 1 (Fall2026): Financial Document Routing for Accounts Payable

- **Project Objective**: Route scanned documents that reach a finance back office
  (invoices, budgets, forms, letters, memos, emails) to the right team with CLIP, and
  maximize macro F1 and invoice recall by comparing zero-shot classification and a
  linear probe on the frozen embeddings with a raw-pixel baseline
- **Dataset Suggestions**:
  [RVL-CDIP 400 per class](https://huggingface.co/datasets/jinhybr/rvl_cdip_400_train_val_test),
  a small subset of [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) with
  16 document classes
  - Keep the six finance-relevant classes `invoice`, `budget`, `form`, `letter`,
    `memo`, and `email`
  - Sample about 100 train images per class, so that the notebooks run in the
    container
- **Tasks**:
  - **Preprocess the Documents**: Load the six classes, sample a class-balanced train
    set, convert the images to RGB, and keep the provided test split
  - **Define the Routing Problem**: Write one text prompt per class (e.g., "a scanned
    invoice from a supplier") and map each class to a destination team (e.g., invoice
    to accounts payable, budget to financial planning, form to compliance)
  - **Route with CLIP**: Compute image and text embeddings with `CLIPModel` and
    `CLIPProcessor`, route zero-shot to the closest prompt, and train a
    `LogisticRegression` linear probe on the image embeddings
  - **Evaluate the Router**: Compute macro F1, invoice recall, and the confusion
    matrix for both routers and for a `LogisticRegression` on 64x64 grayscale pixels,
    and plot macro F1 against the number of labeled images per class
  - **Visualize the Results**: Project the image embeddings to 2D with t-SNE colored
    by class, and show a grid of misrouted documents with their predicted prompt
- **Bonus Ideas (Optional)**: Add receipts from
  [SROIE](https://huggingface.co/datasets/darentang/sroie) as a seventh class and
  measure its zero-shot recall; add an OCR or document model (e.g., LayoutLM, Donut)
  after the router to extract the invoice total, and measure how much the CLIP
  prefilter reduces the number of documents processed

### Milestones

- Milestone 1: Set up the container and the document sample
  - Project tasks: Preprocess the Documents
  - Result: project dir created and container running, and a class-balanced sample of
    the six finance document classes stored as a table of image paths and labels
- Milestone 2: API notebook
  - Project tasks: Route with CLIP
  - Result: `clip.API.ipynb` covering `CLIPProcessor`, image and text embeddings, and
    zero-shot scoring with prompts on a few sample invoices and budgets
- Milestone 3: Example notebook
  - Project tasks: Define the Routing Problem, Route with CLIP, Evaluate the Router,
    Visualize the Results
  - Result: `clip.example.ipynb` running end to end

## Project 2: Chart Images and Volatility Regimes

- **Project Objective**: Test whether CLIP embeddings of candlestick chart images
  identify high-volatility months better than the trailing volatility of the same
  window, and maximize balanced accuracy on a chronological test set
  - Expect the trailing-volatility rule to be hard to beat, because volatility
    persists, and report the result either way
- **Dataset Suggestions**: Daily prices of about 10 liquid ETFs (e.g., `SPY`, `QQQ`,
  `IWM`, `TLT`, `GLD`, `XLE`, `XLF`, `EEM`, `HYG`, `USO`) from
  [yfinance](https://pypi.org/project/yfinance/), rendered to images with
  [mplfinance](https://pypi.org/project/mplfinance/)
  - Optionally compare the regime labels with
    [FRED - VIX](https://fred.stlouisfed.org/series/VIXCLS)
- **Tasks**:
  - **Render the Charts**: Download daily prices and render one 60-day candlestick
    chart every 21 trading days per ETF, hiding the axes, dates, and price levels so
    that the year cannot be read from the image
  - **Define the Label**: Label a chart "high volatility" when the realized
    volatility of the next 21 trading days is above the median of the train period of
    that ETF, and split by date with a 21-day gap between train and test
  - **Classify with CLIP**: Score each chart zero-shot with the prompts "a calm,
    low-volatility price chart" and "a turbulent price chart with large swings",
    train a `LogisticRegression` on the image embeddings from `CLIPModel` and
    `CLIPProcessor`, and retrieve the 10 most similar train charts to vote on the
    regime
  - **Evaluate Against Baselines**: Compute balanced accuracy and ROC AUC with
    bootstrap confidence intervals for the three CLIP methods, a majority-class
    baseline, and a rule based on the trailing 60-day realized volatility
  - **Interpret the Results**: Plot the AUC by year, show the most confident wrong
    charts, and discuss whether the image adds information beyond trailing volatility
- **Bonus Ideas (Optional)**: Prompt textbook patterns (e.g., "head and shoulders")
  and test whether the predicted pattern relates to the next-month return; add
  `BTC-USD` to test transfer across assets

## Project 3: Home Price Estimation From Listing Photos

- **Project Objective**: Test whether CLIP embeddings of listing photos improve a
  home price model, as used to value mortgage collateral, beyond bedrooms, bathrooms,
  area, and city, and minimize the MAPE with calibrated 90% prediction intervals
- **Dataset Suggestions**:
  [House Prices and Images - SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal),
  exterior photos of Southern California homes with price, bedrooms, bathrooms, area,
  and city
  - Sample about 3,000 listings, so that the embeddings run on a laptop CPU
  - Download needs a Kaggle account, and the data has no listing date, so read the
    results as a cross-section valuation and not as a forecast
- **Tasks**:
  - **Preprocess the Listings**: Drop rows with missing values or prices outside the
    1st-99th percentile, use `log(price)` as target, and split into train and test
    stratified by city
  - **Fit the Tabular Baseline**: Fit a `GradientBoostingRegressor` on bedrooms,
    bathrooms, area, and city, and report the MAPE and the median absolute error
  - **Extract CLIP Features**: Compute image embeddings with `CLIPModel` and
    `CLIPProcessor`, zero-shot scores from prompt pairs (e.g., "a luxury home with a
    pool" against "a run-down home"), and the 5 most similar train homes as comps
  - **Fit the Combined Models**: Fit a `Ridge` regression on the embeddings alone,
    and a model on the tabular features plus the embeddings or the zero-shot scores,
    and compare the MAPE with the baseline
  - **Quantify the Uncertainty**: Compute bootstrap 95% confidence intervals of the
    MAPE gain over the baseline, build split-conformal 90% prediction intervals, and
    report the coverage by city and by price tercile
  - **Visualize the Comps**: Plot predicted against actual prices, and show a grid of
    test homes next to their retrieved comps with the price of each
- **Bonus Ideas (Optional)**: Test whether a zero-shot "luxury" score explains the
  residual of the tabular baseline; compare the frozen embeddings with a fine-tune of
  the last layer of the vision encoder
