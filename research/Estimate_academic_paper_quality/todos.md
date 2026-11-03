- [x] Task 1: Separate the code to perform the resolution into a single executable
  that take CSV as inputs and outputs

# Task 2
- [x] When running
  research/Estimate_academic_paper_quality/resolve_papers.py   --input research/Estimate_academic_paper_quality/paper_list.csv   --output ~/papers/run_2026_10/papers.csv
  copy the input to the destination dir as 00-paper_list.csv

- [x] Use for the intermediate files a JSON file, instead of CSV, unless it's
  really best to use a CSV. JSON files are easier to read by humans and machines

- [x] Rename the files created in the output using numbers a 00-paper_list.csv,
  01-resolve.paper_list.json, 02-...
  - The files papers.issues.csv should also be renamed 01-resolve.issues.json
  - do not use the dir stat`state/issues.download.json`

- [x] Name the files after the script that created them using also a number to
  represent the order of the phase in the pipeline (01-, 02-

- [x] In research/Estimate_academic_paper_quality/download_papers.py add a random
  delay to slow down the downloading and avoiding to be detected as a bot

# Task 3
- [ ] Use llm_cli
- [ ] Too many data structures
