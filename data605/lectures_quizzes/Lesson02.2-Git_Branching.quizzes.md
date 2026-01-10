1. What is the primary purpose of branching in Git?
   - A) To delete old files
   - B) To diverge from the main development line **(correct)**
   - C) To merge code without conflicts
   - D) To permanently remove changes
   - E) To manage user permissions

2. What command is used to create a new branch in Git?
   - A) git start <branch-name>
   - B) git new <branch-name>
   - C) git checkout -b <branch-name> **(correct)**
   - D) git branch create <branch-name>
   - E) git add branch <branch-name>

3. When using `git checkout`, what is the effect on the working directory?
   - A) Changes the commit message
   - B) Switches to a different repository
   - C) Moves `HEAD` pointer and updates the files to match the branch **(correct)**
   - D) Deletes the current branch
   - E) Merges branches together

4. What is a fast-forward merge in Git?
   - A) Merging two branches with divergent history
   - B) A merge that happens without conflict markers
   - C) A merge where Git moves the branch pointer forward **(correct)**
   - D) Merging branches that have been deleted
   - E) A type of merge that creates a new branch

5. Why are merge conflicts created in Git?
   - A) When two branches contain deleted files
   - B) When two branches modify the same file **(correct)**
   - C) When commits are pushed out of order
   - D) When branches are deleted accidentally
   - E) When a branch cannot be merged fast forward

6. What is the purpose of `git rebase`?
   - A) To merge two branches with conflicts
   - B) To reset the current branch to a previous commit
   - C) To create a new branch without committing changes
   - D) To apply commits from one branch onto another in a linear fashion **(correct)**
   - E) To delete old commits

7. What is the main difference between rebase and merge?
   - A) Rebase combines multiple histories, merge keeps all histories **(correct)**
   - B) Merge can be performed on any branch, rebase cannot
   - C) Rebase creates a merge commit, merge does not
   - D) They both do the same thing
   - E) Merge affects only local branches, rebase affects remote branches

8. What does the command `git push origin <branch-name>` accomplish?
   - A) It pulls changes from the remote branch
   - B) It creates a new remote branch **(correct)**
   - C) It pushes local changes to the specified remote branch
   - D) It merges two branches locally
   - E) It fetches updates from the remote branch

9. In which scenario should you avoid using rebase?
   - A) When working on a branch only you use **(correct)**
   - B) When pushing to a remote branch shared with others
   - C) When you want to simplify branch history
   - D) When the branch contains many commits
   - E) When merging with the master branch

10. What does a merge commit do in Git?
    - A) It deletes a branch
    - B) It creates a linear history
    - C) It records the point where two branches are combined **(correct)**
    - D) It resets the pointer to a previous commit
    - E) It automatically resolves file conflicts

11. What is the purpose of Git stashing?
    - A) To permanently delete untracked files
    - B) To save changes that are not yet ready to be committed **(correct)**
    - C) To merge branches
    - D) To reset the repository
    - E) To rerun past commands

12. Why should you avoid rebasing commits that have been pushed to a shared repository?
    - A) It will speed up your workflow
    - B) It can create more complicated history
    - C) It will trash the repository
    - D) Others may have based their work on those commits **(correct)**
    - E) It requires altering commit messages

13. Which command is used to list local branches?
    - A) git branches
    - B) git list-branches
    - C) git show-branches
    - D) git branch **(correct)**
    - E) git ls-branch

14. What happens when you perform `git merge` between two branches with no conflicts?
    - A) A merge commit is created **(correct)**
    - B) The files are deleted
    - C) All history is lost
    - D) Only one branch is kept
    - E) It does nothing

15. What is a tracking branch in Git?
    - A) A branch that is always deleted
    - B) A local branch that keeps a reference to a remote branch **(correct)**
    - C) A branch that can only be modified by one user
    - D) A branch that is used for testing
    - E) A branch that does not exist in the local repository

16. When should you create a hotfix branch?
    - A) When merging changes from master
    - B) When a quick bug fix needs to be implemented alongside ongoing work **(correct)**
    - C) When the main branch is inactive
    - D) When testing features
    - E) When deleting features

17. Which command allows you to apply a single commit from one branch onto another?
    - A) git cherry-pick <commit> **(correct)**
    - B) git merge <commit>
    - C) git apply <commit>
    - D) git sync <commit>
    - E) git fetch <commit>

18. How does Git handle merge conflicts?
    - A) Automatically resolves them
    - B) Generates conflict markers and pauses for resolution **(correct)**
    - C) Deletes the conflicting branch
    - D) Merges only certain branches
    - E) Ignores them entirely

19. What do you do after you’ve resolved merge conflict markers in files?
    - A) Commit without further actions
    - B) Remove the conflicting branches
    - C) Rebase the branch
    - D) Use `git add` to mark conflicts as resolved **(correct)**
    - E) Push the branch without changes

20. What is the primary benefit of using `git pull --rebase`?
    - A) It creates merge commits for clarity
    - B) It preserves the branch history of concurrent work **(correct)**
    - C) It merges branches with conflicts
    - D) It allows multiple developers to push simultaneously
    - E) It prevents any changes to the local branch