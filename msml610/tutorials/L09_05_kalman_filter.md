# Implement Notebook Script

notebook = `tutorials/FilterPy/FilterPy.API.ipynb`
script = `tutorials/FilterPy/FilterPy.API_script.md`

## Step 1) Build the Notebook Script
- Come up with a 
- Execute `docs/ai_prompts/notebooks.create_visual_script.md` to create the script
  for the concept

Linear Kalman Filters
Extended Kalman Filter
Unscented Kalman Filter
Ensemble Kalman Filter

## Build the Notebook
- If there is no $notebook
  ```bash
  > cp ./msml610/tutorials/template.ipynb $notebook
  > git add $notebook
  ```
- Execute `docs/ai_prompts/notebooks.implement_script.md` for cells between
  <start> and <end> of $script
- Update $notebook
