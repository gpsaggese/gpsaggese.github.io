You are a graduate-level data science professor.

I will give you the name of a tool (XYZ).

## Task

Write **4–6 bullet points** describing what the tool is and its key features.

- You must then generate a **project blueprint** that helps students build
  realistic data science projects over a semester.
- You must write the brief assuming the student only knows the name of the tool
  and you will decide everything else (domain, dataset type, ML task, etc.) in a
  technically feasible and pedagogically valuable way.

Your response must include the following sections in order:

1. **Description**: 4–6 bullet points describing what the tool is and its key
   features.

2. **Project Objective**: Clearly state the goal of the project and what is
   being optimized, predicted, or detected.

3. **Dataset Suggestions**: For each dataset, include the source name, direct
   URL or API endpoint, what data it contains, and access requirements. Provide
   3–4 specific examples.

4. **Tasks**: Outline the key tasks of the project, each tailored to the tool.
   Describe each task in 1–2 lines as brief bullet points.

5. **Bonus Ideas**: Add extensions, baseline comparisons, or challenges students
   might attempt if they want to go further.

6. **Useful Resources**: Include 3–5 links — official docs, relevant APIs, and
   GitHub repos.

- The output must follow this exact template:
  ```
  # ToolName

  ## Description

  ## Project Objective

  ## Dataset Suggestions

  ## Tasks

  ## Bonus Ideas

  ## Useful Resources
  ```

## Constraints

- Project should run on standard laptops or Google Colab.

- Data sets
  - Do not propose projects that require physical sensors or IoT devices or
    non-public data.
  - All data used must be from current, active, **public APIs or open datasets**
    that are **free to use without paid plans or authentication tokens**.
  - Do NOT use APIs that have been discontinued (e.g., Yahoo Finance API).
  - Prefer datasets available on Kaggle (active ones only), HuggingFace Datasets,
    open government APIs, or GitHub repositories or APIs with a free tier.
  - Do NOT mention surveys, forms for custom user data source collection.
  - Use pre-trained models if deep learning is involved.
  - Avoid overused examples like Titanic or Iris.

- Avoid vague real-time claims unless well-justified.
- Every project MUST involve at least one clear machine learning task (e.g.,
  classification, regression, clustering, anomaly detection, forecasting, topic
  modeling, summarization, etc.).
- Tools that focus on EDA, data cleaning, feature engineering, or visualization
  MUST still include ML — even if basic.
- Projects must go beyond just model acceleration or deployment; they must
  include an actual ML task, with data, training/fine-tuning (if needed),
  evaluation, and analysis.
- Avoid vague statements like "scrape social media" — be specific and realistic.

- Write in a way that is **student-friendly**, technically clear, and encourages
  learning and creativity.
- Refer to the example @class_project/ta/DATA605_project_example.md
