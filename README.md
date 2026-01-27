# GP's University of Maryland Machine Learning Classes

![alt text](https://1000logos.net/wp-content/uploads/2022/07/University-of-Maryland-Logo.png)

- DATA605: Big Data Systems
- MSML610: Advanced Machine Learning

# Cloning the GitHub class repo

- Clone the GitHub Class Repository to get started:
  ```
  > git clone git@github.com:gpsaggese/umd_classes.git
  ```
- More detailed instructions are in each project dir

# Conventions
- We indicate the execution of an OS command (e.g., Linux / MacOS) from the terminal
  of your computer with:
  ```
  > ... Linux command ...
  ```
  E.g.,
  ```
  > echo "Hello world"
  Hello world
  ```

- We indicate the execution of a command inside a Docker container with:
  ```
  docker> ls 
  ```

- We indicate the execution of a Postgres command from the `psql` client with:
  ```
  psql> 
  ```

# Office hours
- Contact: gsaggese@umd.edu

# How to contribute

Contributions to the repository are done using the Fork and PR method. The steps are:

1. Create an Issue
2. Fork the repository
3. Create a new branch
4. Make your changes
5. Create a pull request from your branch to the main repository
6. Wait for the pull request to be reviewed and merged

For more information about Forks, see the [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks).

**1- Create an Issue**

Create an issue to discuss the changes you want to make. Keep record of the issue number generated.

**2- Fork the repository**

![Fork](images/2-create-fork.png)

Fork will create a copy of the repository in your GitHub account. This will allow you to make changes to the repository without affecting the original one. Commits can be merged back to the original repository by creating a pull request.

This method is useful to reduce the noise created by multiple pull requests and commits in the main repository.



**3- Create a new branch on your forked repository**

Create a new branch on your forked repository to make your changes. Include the issue number in the branch name.

```
# Always clone your forked repository, not the original one.
> git clone git@github.com:{your_username}/umd_classes.git umd_classes
> cd umd_classes
> git checkout -b TutorTask{issue_number}_{project_branch_name}
```

**Note:** Always reference the issue number in the branch name.

**4- Make and commit your changes**

Make your changes to the code in the new branch. The commit message should include a reference to the issue number.

```
> git add .
> git commit -m "{whatever commit message you want} (gpsaggese/umd_classes#{issue_number})"
> git push origin TutorTask{issue_number}_{project_branch_name}
```

**Note:** The prefix `gpsaggese/umd_classes` is required to link the commit to an issue in the original repository. If the issue is in your forked repository, this isn't required.

**5- Create a pull request from your branch to the main repository**

Create a pull request from your branch to the main repository. The pull request should include a reference to the issue number.

Including this text in the pull request description will automatically close the issue once the pull request is merged.

```
Fixes gpsaggese/umd_classes#{issue_number}
```

For more information about how to reference an issue in a pull request, see the [GitHub Docs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

**6- Wait for the pull request to be reviewed and merged**

Wait for the pull request to be reviewed and merged. Add the expected reviewer as the `assignee`.



