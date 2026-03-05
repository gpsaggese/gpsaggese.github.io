# Summary

This document is a reference for future TAs when preparing project descriptions
for student capstone projects. It covers the end-to-end process from tool list
creation to student assignment and onboarding.

## Tool List Creation

- TAs create a list of all possible tools they find interesting that are related
  to the course
  - TAs can look at tools released for capstone projects in past courses
- GP reviews the list and provides feedback to TAs
- The steps above are repeated until the tool list is finalized

## Description Generation

- TAs run the
  [generate_class_project_description.py](https://github.com/gpsaggese/umd_classes/blob/master/class_project/ta.class_project_gen/generate_class_project_description.py)
  script to generate descriptions for all tools in the list:
  ```bash
  > python class_project/ta/generate_class_project_description.py \
      --input class_project/DATA605/Spring2026/projects.csv \
      --out_dir class_project/DATA605/Spring2026/projects_descriptions \
      --max_projects 2
  ```
- TAs review all descriptions to ensure they align with the requirements and
  difficulty levels
- TAs can tweak the prompt for description generation if necessary

## Tool Release and Student Assignment

- GP releases links to tool list and descriptions for students
- Students fill the Google signup form mentioning their two choices
- TAs assign tools to students:
  - Priority is given to groups first, then individual students
  - In case of collisions/overlaps between choices, try to resolve by looking at
    both choices
  - TAs send emails to all students/groups whose choices overlap to choose new
    tools
  - Resolve all conflicts until no overlaps/collisions remain
  - TAs send regular emails to students who did not fill the Google signup form
    to push them to make their selections
  - TAs release the list of available tools through ELMS announcements after the
    first round of tool assignment
    - This helps students who have not filled the form see which tools are
      available and taken, reducing chances of more collisions

## Final Assignment and Onboarding

- TAs release the final list of tools assigned to students through ELMS
  announcements
- TAs release a small assignment to push students to set up Git and Docker
  environments on their system
- TAs use the
  [invite_github_collaborators.py](https://github.com/gpsaggese/umd_classes/blob/UmdTask89_Update_Github_Invite_Collaborators_script/class_project/ta.class_project_gen/invite_github_collaborators.py)
  script to send out GitHub collaborator invites to all students
  - To create a token for the repo, two options are available:
    - **Option A (recommended)**: fine-grained PAT
      - GitHub -> Settings -> Developer settings -> Personal access tokens ->
        Fine-grained -> Generate
      - Resource owner: pick the org (or your account if it is under you)
      - Repository access: Only selected repositories -> choose the target repo
      - Repository permissions: Administration -> Write
    - **Option B**: classic PAT
      - GitHub -> Settings -> Developer settings -> Personal access tokens ->
        Tokens (classic) -> Generate
      - Scopes: check `repo` (covers private repos and includes `repo:invite`)
- TAs create issues for all assigned tools using the script (TODO: add link to
  script)
- TAs inform students to self-assign themselves to the respective issues
