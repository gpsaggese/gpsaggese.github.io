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

- Usual tutorial "Learn CLIP in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Create `tutorials/CLIP/` with `.claude/skills/tutorials_in_60_mins.create/SKILL.md`
  - A previous session delivered a CLIP project: see the `Result` column in
    `class_project/project_descriptions/README.md`, and reuse what is good
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `clip_utils.py`
  - `clip.API.ipynb`
  - `clip.example.ipynb`

# Project

## Project 1: Back-Office Document Routing

- **Difficulty**: 2 (Medium)
- **Project Objective**: Route scanned back-office documents (e.g., invoices, forms,
  memos, letters) into categories with CLIP, and maximize routing accuracy by
  comparing zero-shot classification with a linear probe on the frozen embeddings
- **Dataset Suggestions**:
  [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) (16 document classes,
  used as a proxy for back-office documents)
  - Sample about 100 images per class, so that the notebooks run in the container
- **Tasks**:
  - **Preprocess the Documents**: Stream a class-balanced sample of RVL-CDIP, convert
    the images to RGB, and split them into train and test sets
  - **Define the Routing Problem**: Write one text prompt per class (e.g., "a scanned
    invoice") and fix the label set that the router must predict
  - **Route with CLIP**: Compute image and text embeddings with `CLIPModel` and
    `CLIPProcessor`, route zero-shot to the closest prompt, and train a
    `LogisticRegression` linear probe on the image embeddings
  - **Evaluate the Router**: Compute accuracy, macro F1, and the confusion matrix for
    both routers, and plot accuracy against the number of labeled images per class
  - **Visualize the Results**: Project the image embeddings to 2D with t-SNE colored
    by class, and show a grid of misrouted documents with their predicted prompt
- **Bonus Ideas (Optional)**: Add an OCR or document model (e.g., LayoutLM, Donut)
  after the router and measure how much the CLIP prefilter improves throughput or
  accuracy; compare several prompt templates

### Milestones

- Milestone 1: Set up the container and the document sample
  - Project tasks: Preprocess the Documents
  - Result: `tutorials/CLIP/` container running, and a class-balanced RVL-CDIP sample
    stored as a table of image paths and labels
- Milestone 2: API notebook
  - Project tasks: Route with CLIP
  - Result: `clip.API.ipynb` covering `CLIPProcessor`, image and text embeddings, and
    zero-shot scoring with prompts on a few sample images
- Milestone 3: Example notebook
  - Project tasks: Define the Routing Problem, Route with CLIP, Evaluate the Router,
    Visualize the Results
  - Result: `clip.example.ipynb` running end to end

## Project 2: Generative Art From Text Prompts

- **Difficulty**: 2 (Medium)
- **Project Objective**: Utilize CLIP-ViT-Large-Patch14 to generate artistic images
  based on user-defined text prompts, optimizing the creativity and relevance of the
  generated images
- **Dataset Suggestions**: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart),
  a diverse collection of artworks categorized by style, artist, and genre
- **Tasks**:
  - **Set Up the CLIP Model**: Load the CLIP model and required libraries for image
    generation
  - **Design Text Prompts**: Create a system for users to input creative text prompts
    for generating art
  - **Generate Images**: Implement a pipeline that generates images based on the text
    prompts using CLIP's capabilities
  - **Assess Quality**: Develop a mechanism to evaluate the quality and relevance of
    generated images through user feedback or similarity metrics
  - **Showcase Results**: Create a web app or dashboard to display generated artworks
    alongside input prompts
- **Bonus Ideas (Optional)**: Allow users to refine prompts iteratively and analyze
  how changes affect the generated art

## Project 3: Multimodal Sentiment Analysis on Social Media Posts

- **Difficulty**: 3 (Hard)
- **Project Objective**: Implement a multimodal sentiment analysis system using
  CLIP-ViT-Large-Patch14 to analyze social media posts that include both images and
  text, optimizing for sentiment classification accuracy
- **Dataset Suggestions**:
  [MVSA-Multiple](https://www.kaggle.com/datasets/vincemarcs/mvsamultiple), tweets
  with an image, a text, and a sentiment label
- **Tasks**:
  - **Ingest the Data**: Collect and preprocess tweets along with their associated
    images from the dataset
  - **Extract Features**: Use CLIP to extract features from both text and images for
    each post
  - **Classify Sentiment**: Train a classifier (e.g., logistic regression or neural
    network) using the extracted features to predict sentiment
  - **Evaluate the Model**: Evaluate the model's performance using accuracy,
    precision, and recall, and visualize the results
  - **Analyze the Results**: Analyze the influence of image content on sentiment
    classification and present findings in a report
- **Bonus Ideas (Optional)**: Explore the impact of different image types (memes,
  infographics) on sentiment prediction accuracy
