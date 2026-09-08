# How to Contribute

## Conventions
- We indicate the execution of an OS command (e.g., Linux / macOS) from the
  terminal of your computer with:
  ```bash
  > ... Linux command ...
  ```

  E.g.,
  ```bash
  > echo "Hello world"
  Hello world
  ```

## Overview
- Contributions to the repository are done using the Fork and PR method. The
  steps are:
  1. Create an Issue
  2. Fork the repository
  3. Clone your fork and sync it with upstream
  4. Create a new branch on your forked repository
  5. Make and commit your changes
  6. Create a pull request from your branch to the main repository
  7. Wait for the pull request to be reviewed and merged

- For more information about forks, see the
  [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks).

## Prerequisites
- Before contributing, make sure you have:
  - A [GitHub account](https://github.com)
  - Git installed on your machine
  - SSH keys configured for GitHub (see
    [GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))
  - Basic familiarity with Git (clone, branch, commit, push)

### Create an Issue
- Create an issue to discuss the changes you want to make. Keep a record of the
  issue number, you will reference it in your branch name, commit messages, and
  pull request.

### Fork the Repository
![Fork](../assets/images/2-create-fork.png)

- A fork creates a copy of the repository in your GitHub account. This allows
  you to make changes without affecting the original repository. Changes can be
  merged back by creating a pull request.

- This approach reduces noise from multiple commits and branches in the main
  repository.

### Clone Your Fork and Sync It With Upstream
- Clone your forked repository (not the original one) and add the original
  repository as an `upstream` remote so you can keep your fork up to date:
  ```bash
  # Always clone your forked repository, not the original one.
  > git clone --recursive git@github.com:{your_username}/umd_classes.git umd_classes
  > cd umd_classes
  > git remote add upstream git@github.com:gpsaggese/umd_classes.git
  > git fetch upstream
  > git checkout master
  > git merge upstream/master
  ```

- Repeat the `fetch` / `checkout master` / `merge` steps any time before
  starting new work, to make sure you branch off the latest code.

### Create a New Branch on Your Forked Repository
- Create a new branch that includes the issue number. For example, for issue
  #42:
  ```bash
  > git checkout -b UmdTask{issue_number}_{short_description}
  ```

- Example branch name: `UmdTask42_Add_Postgres_Tutorial`

**Note:** Always include the issue number in the branch name.

### Make and Commit Your Changes
- Make your changes on the new branch. Stage and commit them with a message that
  references the issue number:
  ```bash
  > git add {file1} {file2}
  > git commit -m "{commit message} (gpsaggese/umd_classes#{issue_number})"
  > git push origin UmdTask{issue_number}_{short_description}
  ```

- The prefix `gpsaggese/umd_classes` is required to link the commit to an issue
  in the original repository. If the issue is in your forked repository, this
  prefix is not required.

- Prefer staging specific files (`git add {file}`) rather than `git add .` to
  avoid accidentally including unintended changes.

### Create a Pull Request From Your Branch to the Main Repository
- Open a pull request from your branch to the main repository. Include the
  following line in the pull request description to automatically close the
  issue once the PR is merged:
  ```verbatim
  Fixes gpsaggese/umd_classes#{issue_number}
  ```

- The `Fixes` keyword must appear at the start of a line to trigger auto-close.

- For more information about linking a pull request to an issue, see the
  [GitHub Docs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

### Wait for the Pull Request to Be Reviewed and Merged
- Assign the expected reviewer under the **Reviewers** field (not the Assignees
  field) in the pull request. Wait for the review and address any requested
  changes before the PR is merged.

- If conflicts arise between your branch and the main branch, sync your fork
  (Step 3) and rebase or merge `master` into your branch before requesting a
  review.
