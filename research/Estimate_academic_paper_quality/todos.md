- For the code in research/Estimate_academic_paper_quality/
  - implement the requested task
  - update / add unit tests
  - update the documentation in research/Estimate_academic_paper_quality/doc

# Task 1
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
- [x] Use llm_cli

# Task 4
- [x] For each stage, improve output by writing:
  - What output files were generated
  - Where is the file with the issues generated
  - How many papers were processed
  - How many papers were completed successfully

# Task 5
- Print how many should be processed and how many are generated
- [ ] Add a switch --max_num to control the maximum number of papers processed in a
stage

# Task 6
- [x] What is semanticscholar

# Task 7
- Populate /Users/saggese/src/umd_classes1/research/Estimate_academic_paper_quality/affiliation_tiers.yaml

# Task 8
- [x] Convert /Users/saggese/papers/run_2026_10/04-report.issues.csv in a json like for
  the other stages

> ls /Users/saggese/papers/run_2026_10/*issues*
/Users/saggese/papers/run_2026_10/01-resolve.issues.json  /Users/saggese/papers/run_2026_10/03-extract.issues.json  /Users/saggese/papers/run_2026_10/04-report.issues.csv
/Users/saggese/papers/run_2026_10/02-download.issues.json /Users/saggese/papers/run_2026_10/04-rate.issues.json

# Task 0

- [ ] Too many data structures
