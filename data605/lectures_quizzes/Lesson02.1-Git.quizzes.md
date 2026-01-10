1. What is the main purpose of branching in Git?
   - A) To delete code
   - B) To simplify commit messages
   - C) To create a new repository
   - D) To work without affecting the main code **(Correct answer)**
   - E) To automatically merge files

2. Which command creates a new branch in Git?
   - A) git new branch
   - B) git branch -n
   - C) git create branch
   - D) git branch <branch-name> **(Correct answer)**
   - E) git checkout -b

3. What does the `HEAD` pointer indicate in Git?
   - A) The main branch
   - B) The current branch **(Correct answer)**
   - C) The last commit
   - D) A remote repository
   - E) The commit history

4. When you use `git checkout <branch-name>`, what happens?
   - A) The new branch is created
   - B) You switch to the specified branch **(Correct answer)**
   - C) Changes are permanently deleted
   - D) The repository is pushed to remote
   - E) All files are reverted to their original state

5. What occurs during a fast-forward merge?
   - A) A new merge commit is created
   - B) The branch pointer moves forward without a commit **(Correct answer)**
   - C) Conflicts must be resolved manually
   - D) The branch is deleted
   - E) The branch history is rewritten

6. What is the main advantage of using `git rebase` instead of `git merge`?
   - A) It makes the commit history more linear **(Correct answer)**
   - B) It prevents any conflicts
   - C) It automatically merges all branches
   - D) It deletes the original branches
   - E) It requires more branches

7. Which of the following is true about merging conflicts?
   - A) Git automatically resolves all conflicts
   - B) Conflicts must be resolved by the user **(Correct answer)**
   - C) Merging conflicts are logged automatically
   - D) Conflicts can always be avoided
   - E) Git creates a separate repository for conflicts

8. What does `git cherry-pick <commit>` do?
   - A) Deletes a commit
   - B) Applies a single commit from another branch **(Correct answer)**
   - C) Reverts the last commit
   - D) Creates a new branch
   - E) Merges all branches

9. When is a merge commit created in Git?
   - A) When branches are identical
   - B) When there are divergent histories **(Correct answer)**
   - C) When pushing to a remote repository
   - D) After rebasing
   - E) In a fast-forward scenario

10. What does the command `git log` provide?
    - A) The current working directory
    - B) A list of all remote repositories
    - C) Information about commits **(Correct answer)**
    - D) The status of the working directory
    - E) The number of branches

11. In a Git remote branch, what does the term "tracking branch" refer to?
    - A) A branch that can never be modified
    - B) A local branch that references a remote branch **(Correct answer)**
    - C) A branch that has been deleted
    - D) A branch with permission to push updates
    - E) A branch that cannot be merged

12. What defines the "Integration-Manager Workflow" in Git?
    - A) Everyone has write access to the main branch
    - B) One official repo is maintained by a manager **(Correct answer)**
    - C) Developers cannot fork branches
    - D) Only admins can create branches
    - E) All contributors push directly to the master branch

13. How does `git fetch` work?
    - A) It merges the current branch with the remote
    - B) It updates local tracking branches with changes from the remote **(Correct answer)**
    - C) It deletes old branches
    - D) It creates new branches
    - E) It pushes local changes to remote

14. Why are rebases discouraged on shared branches?
    - A) They are too complex
    - B) They can overwrite commits that others depend on **(Correct answer)**
    - C) They are unnecessary
    - D) They slow down the repository
    - E) They make branches unruly

15. In terms of Git workflow, what is a "topic branch"?
    - A) A main development branch
    - B) A long-term stable branch
    - C) A short-lived branch for a single feature **(Correct answer)**
    - D) A branch only for fixing bugs
    - E) A branch that cannot be merged

16. What does the `git merge` command require to complete a merge successfully?
    - A) No changes in either branch
    - B) Only one branch can exist
    - C) At least one common commit in the histories **(Correct answer)**
    - D) A rebase must be performed first
    - E) The branches must be identical

17. What is the purpose of `git bisect`?
    - A) To merge branches
    - B) To identify commits that introduced bugs **(Correct answer)**
    - C) To rebase branches
    - D) To delete old commits
    - E) To update the working directory

18. What is a major consequence of using `git rebase` on shared branches?
    - A) It simplifies the history
    - B) It can lead to discrepancies in commit history among collaborators **(Correct answer)**
    - C) It resolves all conflicts automatically
    - D) It creates new branches
    - E) It makes the repository larger

19. When doing a pull request (PR), who typically reviews the submitted changes?
    - A) The original author only
    - B) Any member of the repository
    - C) The project maintainer **(Correct answer)**
    - D) An automated script
    - E) The GitHub community

20. In Git philosophy, how does the concept of history differ between using merge versus rebase?
    - A) Merge creates a linear history; rebase is messy
    - B) Rebase creates a linear history; merge reflects actual project evolution **(Correct answer)**
    - C) Both create identical histories
    - D) Merge requires more steps than rebase
    - E) Both are equally preferred for every situation