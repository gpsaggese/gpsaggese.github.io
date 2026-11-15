# Log

- [x] Build EAPQ pipeline end-to-end
  - One shotting it is a terrible idea, since the same problem is everywhere
  - Much better to build / debug one chunk at the time
- [ ] Rename all files as EAPQ
- [.] Debug the process docs/planning/all.plan_work.how_to_guide.md
- [.] Debug pipeline
- [ ] Read all the code top to bottom
  - research/Estimate_academic_paper_quality/code_tour.md
- [x] Get QS key
- [x] Get Semantic Scholar key
- [.] Understand / build tutorial Semantic Scholar
  - tutorials/semantic_scholar/
- [ ] Write paper papers/Causal_Analysis_for_Finance/paper.pdf
- [ ] Extract important claim using LLM

- [ ] Extract graph using DCDS?
  - Move DataMap out, run it according to the paper, clean it up
  - Extract single relationship and build graph
  - How is it different than Theory Store
- [ ] Update slides
  - papers/Causal_Analysis_for_Finance/notes.slides.pdf

# Implement Pipeline

## Define

- Follow `docs/planning/all.plan_work.how_to_guide.md`
  - [x] Write idea.md
  - [x] Generate PRD and improve
    ```
    claude> /project.write_prd <project>/idea.md
    claude> /text.criticize <project>/docs/01-prd.md
    ```
    - Generate `research/Estimate_academic_paper_quality/docs/01-prd.md`
- Generate `research/Estimate_academic_paper_quality/README.md`
- [x] Generate architecture
  ```
  claude> /architecture.write_design <project>/docs/01-prd.md
  claude> /text.criticize <project>/docs/02-design.architecture.md
  ```

- `./helpers_root/.claude/skills/architecture.rules.md`

## Plan
  ```text
  claude> /project.create_plan research/Estimate_academic_paper_quality/docs/02-design.architecture.md
  claude> /auto_task.criticize research/Estimate_academic_paper_quality/docs/04-plan.md
  claude> /auto_task.create_plan_for_feature research/Estimate_academic_paper_quality/docs/04-plan.md PR_A1
  claude> /auto_task.criticize research/Estimate_academic_paper_quality/docs/05-tasks.md
  ```

- [ ] Read open_md.py -i research/Estimate_academic_paper_quality/docs/02.*
- [ ] Read open_md.py -i research/Estimate_academic_paper_quality/docs/03-decisions.md
- [ ] ...
- [ ] Read code top-to-bottom

- What is missing
  - [ ] A document that explains how to run the code
    /readme_file.create

# Idea

- Create a pipeline that accepts a list of links of papers in the form of partial
  titles, links, etc

1) Convert the list of papers into a list with a fixed format (the one in
   .claude/skills/papers.references.md) and downloading links

2) Download the papers and converts them in markdown using download_academic_paper.py

3) Score the quality of each paper using the following criteria
   - Quality of the paper (e.g., institution, authors, publication venue)
   - Reported Sharpe ratio and other performance metrics
   - Recency (publication year)
   - Target asset class (e.g., US equities)
   - Theoretical domain (e.g., micro vs. macro)
   - Availability of the required data in Atlas (e.g., asset coverage, eval vs prod
     data, ...)
   - Nature of the reported mechanism (causal vs. associational)
