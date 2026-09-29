**Description**

CLIP-ViT-Large-Patch14 is a powerful vision-language model developed by OpenAI that
enables understanding and generation of images and text together. This model can be
used for various tasks such as zero-shot classification, image generation from text
prompts, and more.

Technologies Used
CLIP-ViT-Large-Patch14

- Combines visual and textual understanding to perform image-text matching
- Supports zero-shot learning, allowing the model to classify images without explicit
  training on specific categories
- Utilizes a transformer architecture for efficient processing of visual and textual
  data

### Fall2026

#### Tutorial

- Usual tutorial "Learn CLIP in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `clip_utils.py`
  - `clip.API.ipynb`
  - `clip.example.ipynb`

#### Project: Back-Office Document Routing

- **Project Objective**: Many back-office pipelines receive mixed scans: invoices,
  receipts, bank statements, checks, tax forms, contracts. CLIP can route these into
  categories zero-shot or with a light linear probe. OCR or a document model
  (LayoutLM, Donut) then handles extraction. The project is a
  classification-plus-routing pipeline, and you can measure how much a CLIP prefilter
  improves throughput or accuracy.
- TODO(ai_gp): Expand this project with the dataset and the tasks

#### Milestones

- TODO(ai_gp): Add milestones related to the project

### Project 1: Image Classification with Zero-Shot Learning

- **Difficulty**: 1 (Easy)
- **Project Objective**: Leverage CLIP-ViT-Large-Patch14 to classify images from a
  dataset of everyday objects using natural language descriptions, optimizing for
  accuracy in zero-shot classification
- **Dataset Suggestions**: Use the "CIFAR-10" dataset available on Kaggle, which
  contains 60,000 32x32 color images in 10 classes
- **Tasks**:
  - **Set Up the CLIP Model**: Install the required libraries and load the CLIP model
  - **Data Preparation**: Preprocess the CIFAR-10 images and prepare corresponding
    textual labels for each class
  - **Zero-Shot Classification**: Use CLIP to classify images based on the textual
    descriptions without additional training
  - **Evaluation**: Compute accuracy and generate a confusion matrix to evaluate
    classification performance
  - **Visualization**: Create visual representations of the classification results,
    highlighting correct and incorrect predictions
- **Bonus Ideas (Optional)**: Experiment with different text descriptions to see how
  they affect classification accuracy

### Project 2: Generative Art From Text Prompts

- **Difficulty**: 2 (Medium)
- **Project Objective**: Utilize CLIP-ViT-Large-Patch14 to generate artistic images
  based on user-defined text prompts, optimizing the creativity and relevance of the
  generated images
- **Dataset Suggestions**: Use the "WikiArt" dataset available on Kaggle, which
  contains a diverse collection of artworks categorized by style, artist, and genre
- **Tasks**:
  - **Set Up the CLIP Model**: Load the CLIP model and required libraries for image
    generation
  - **Text Prompt Design**: Create a system for users to input creative text prompts
    for generating art
  - **Image Generation**: Implement a pipeline that generates images based on the
    text prompts using CLIP's capabilities
  - **Quality Assessment**: Develop a mechanism to evaluate the quality and relevance
    of generated images through user feedback or similarity metrics
  - **Showcase Results**: Create a web app or dashboard to display generated artworks
    alongside input prompts
- **Bonus Ideas (Optional)**: Allow users to refine prompts iteratively and analyze
  how changes affect the generated art

### Project 3: Multimodal Sentiment Analysis on Social Media Posts

- **Difficulty**: 3 (Hard)
- **Project Objective**: Implement a multimodal sentiment analysis system using
  CLIP-ViT-Large-Patch14 to analyze social media posts that include both images and
  text, optimizing for sentiment classification accuracy
- **Dataset Suggestions**: Use the "Twitter Sentiment Analysis" dataset available on
  Kaggle, which contains tweets labeled with sentiment (positive, negative, neutral)
  and includes images
- **Tasks**:
  - **Set Up the CLIP Model**: Load CLIP and necessary libraries for processing both
    text and images
  - **Data Ingestion**: Collect and preprocess tweets along with their associated
    images from the dataset
  - **Feature Extraction**: Use CLIP to extract features from both text and images
    for each post
  - **Sentiment Classification**: Train a classifier (e.g., logistic regression or
    neural network) using the extracted features to predict sentiment
  - **Model Evaluation**: Evaluate the model's performance using metrics such as
    accuracy, precision, and recall, and visualize the results
  - **Analysis of Results**: Analyze the influence of image content on sentiment
    classification and present findings in a report
- **Bonus Ideas (Optional)**: Explore the impact of different image types (memes,
  infographics) on sentiment prediction accuracy
