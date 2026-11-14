- Read `class_project/prompt.update_projects.md`

- Process the project tables in:
  - `class_project/msml610/Fall2026/class_project.csv`
  - `class_project/data605/Fall2026/class_project.csv`

- Each project in the tables should have a corresponding issue in
  [gpsaggese.github.io issues](https://github.com/gpsaggese/gpsaggese.github.io/issues)
  - The issue name is derived from the project name
  - E.g., `MSML610_Fall2026_How_to_measure_LLM_compliance`

# Fetch the Issues

- Download all the issues, open and closed, with their authors and bodies:
  ```bash
  > gh issue list \
      -R gpsaggese/gpsaggese.github.io \
      --state all \
      --limit 1000 \
      --json number,title,state,body,author
  ```
- Save the output in a scratch file and work from it

# Expected Issue Names

- Build the expected issue name as `<COURSE>_Fall2026_<NAME>`
  - `<COURSE>` is `MSML610` or `DATA605`
  - For a research idea, `<NAME>` is the idea file name without the prefix
    (`draft.`, `in_progress.`) and without `.md`
    - E.g., `draft.Build_an_LLM_Arena.md` -> `MSML610_Fall2026_Build_an_LLM_Arena`
  - For a tutorial, `<NAME>` is the tool name
    - E.g., `Luigi_Project_Description.md` -> `DATA605_Fall2026_Luigi`
  - For a tutorial with a project number, append `_Project<N>`
    - E.g., `DoWhy_Project_Description.md#project-3-...` ->
      `MSML610_Fall2026_DoWhy_Project3`

# Match Projects to Issues

- Match each distinct project in the column `Chosen Project (name)` to one issue
- Match by name first
  - Ignore case, prefixes, `.md`, and the suffix `_Project_Description`
- If the name is different, match by team
  - Look for the `GitHub Username` of the team members in the issue body and in the
    issue author
  - Confirm that the issue refers to the same project description or research idea
  - Check the issues whose name does not match but whose body does, since the team
    might have used a different title
- If there are many issues for the same team, choose in this order
  - The open issue
  - The issue whose name is equal to the expected name
  - Report the others as duplicates
- Accept a closed issue only if it is the single match
- Do not use issues from other semesters (e.g., `Fall2025_*`, `DATA605_Spring2026_*`)

# Update the Tables

- Add or modify these two columns, in this order, as the last columns of each table
  - `GitHub Issue`: the full URL of the issue, e.g.,
    `https://github.com/gpsaggese/gpsaggese.github.io/issues/882`
  - `GitHub Issue (name)`: the title of the issue, e.g.,
    `MSML610_Fall2026_How_to_measure_LLM_compliance`
  - Leave both cells empty if there is no issue
  - Fill all the rows of the same team with the same values
  - Take the title from the downloaded issues, so that it is not typed by hand
- Use the `csv` module of Python and not `csv.DictReader`, since the header has
  repeated column names (e.g., `Choice 1`)
- Preserve the line terminator (CRLF) and the quoting of the original file
  - Check that reading and writing the file without changes gives identical bytes

# Report the Missing Issues

- Report the projects that don't have a corresponding issue
  - Print the course, the group, and the project name
  - Print the expected issue name
- For each missing issue, look for an issue with a misspelled or different name
  - Compare the expected name with the titles of all the issues with a fuzzy match
    (e.g., `difflib`), ignoring stop words
  - Search the bodies and the authors of all the issues for the GitHub usernames of
    the team and for key words of the project
  - Report the issue found, if any, and say if it is the same team and the same
    project
  - Do not fill the table with an issue found in this way until the user confirms
- Write the report in `class_project/gp/report.missing_gh_issues.md`
  - For each missing project, list the students with their name and email
  - Use the column `Personal Email`
  - If there is no personal email, use the UMD login ID from `Login ID`
  - Remove pronouns from the names

# Check the GitHub Project

- Fetch the items of the GitHub project
  [projects/8](https://github.com/users/gpsaggese/projects/8/views/1):
  ```bash
  > gh project item-list 8 --owner gpsaggese --format json --limit 1000
  ```
- Report the issues in the tables that are not in the project
- Report also the issues in the project that are not in the tables
