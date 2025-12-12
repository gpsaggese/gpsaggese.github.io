# **MSML610**

# **Github and Docker Setup Assignment**

## **Objective:**

To help you set up and understand the basic GitHub and Docker commands. This will ensure you can focus on your projects rather than getting stuck on environment or tooling issues.

## **Instructions for Submission**

For each step below, take a screenshot of the successful execution.  
Collect all screenshots into a single PDF file.  
Add brief comments/notes and headers for each screenshot explaining what the step does.  
Submit the PDF on ELMS.  
File Name convention should be \*\*Git\_Docker\_Setup\_UID\_FirstName\*\*

## **Follow the structure as outlined below.**

1. ## **Set up GitHub SSH access and personal access token (PAT)**

   ## To generate a new SSH key, follow the official GitHub [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

     
   Ensure that you save the SSH key with the below name format and at the specified location  
   1. File location: \`\`\` \~/.ssh/id\_rsa.causify-ai.github \`\`\`  
   2. 

          	Example command to generate SSH key:

   

   ```` ```CODE ````        

   `ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_rsa.causify-ai.github`

   

   To create a Personal Access Token (classic) with necessary scopes like repo, workflow, etc., click this [link](https://github.com/settings/tokens)  and then click "Generate new token (classic)".

   After obtaining the token, store it in a file named   *github\_pat.causify-ai.txt* at the specified path

   

          File location: \`\`\` \~/.ssh/github\_pat.causify-ai.txt \`\`\`

   

          Example command to save using vim:

   

   ```` ```CODE ````

              `vim ~/.ssh/github_pat.causify-ai.txt`

2. ## **Clone the repo**

```` ```CODE ````  
`git clone --recursive git@github.com:gpsaggese-org/umd_classes.git ~/src/umd_classes1`

*NOTE: The above command might not work sometimes, in which case try the alternative command using HTTP instead of SSH:*

```` ```CODE ````  
`git clone --recursive https://github.com/gpsaggese-org/umd_classes.git ~/src/umd_classes1`

If you’re facing issues in cloning the repo (and you get errors such as **Permission denied (publickey)), try running the below step first and then come back to run this step**

3. ## **Build the thin environment**

     
   Create the "thin environment" (or venv) which contains the minimum set of dependencies needed for running Docker.  
     
   Build the thin environment; this is done once per client  
     
      	If you are in the \`\`\`umd\_classes\`\`\` repo  
     
   ```` ```CODE ````  
      `/dev_scripts_umd_classes/thin_client/build.py`  
     
      	Otherwise:  
     
   ```` ```CODE ````  
      `> cd umd_classes`  
      `> ./dev_scripts_umd_classes/thin_client/build.py`  
     
   `NOTE: If the above commands do not work, try the below command`  
     
   ```` ```CODE: ````  
   `> ./helpers_root/dev_scripts_helpers/thin_client/build.py`  
   

4. ## **Activate the environment**

     
   Activate the thin environment; **make sure it is always activated.**  
     
     
   ```` ```CODE ````  
   `> source dev_scripts_umd_classes/thin_client/setenv.sh`  
     
5. **Install and test Docker**  
     
   Get familiar with [Docker](https://docs.docker.com/get-started/overview/) if you are not. We will work in a Docker container that has all the required dependencies installed.  
     
   **Supported OS**  
     
   For Linux Ubuntu:  
   1. Operating System: Ubuntu 24.04 LTS  
   2. Docker: v28

   Using Windows with WSL has always been tricky and unpredictable.

   3. If you are using Windows, we suggest to use dual boot with Linux

   

   You can use PyCharm / VSCode on your laptop to edit code, but you want to run code inside the dev docker container since this makes sure everyone is running with the same system, and it makes it easy to share code and reproduce problems

   

   **How to Install Docker Desktop on your PC**

   

* [Links](https://docs.docker.com/engine/install/):  
  * [Mac](https://docs.docker.com/desktop/install/mac-install/)  
  * [Linux](https://docs.docker.com/desktop/install/linux-install/)


  For Mac you can also install docker-cli without the GUI using


  

  ```` ```CODE ````

  `> brew install docker`

  `> brew link docker`

  `> brew install colima`


6. **Checking Docker installation**  
     
   Check the installation by running:  
     
   \`  
   ``` ``CODE ```  
   `> docker pull hello-world`  
   `Using default tag: latest`  
   `latest: Pulling from library/hello-world`  
   `Digest: sha256:fc6cf906cbfa013e80938cdf0bb199fbdbb86d6e3e013783e5a766f50f5dbce0`  
   `Status: Image is up to date for hello-world:latest`  
   `docker.io/library/hello-world:latest`  
     
     
     
     
   **Add user to Docker group and sudoers (ONLY for Linux users, not Mac)**  
     
   1. This section only applies to Linux, macOS users can skip this. Docker Desktop handles permissions automatically on Mac.

   

   2. Add your user to the docker group to run Docker without sudo

   

   

   ```` ```CODE ````

   `> sudo usermod -aG docker $USER`

   

   Restart your shell session (log out and log back in), or run:

   

   ```` ```CODE ````

   `> newgrp docker`

   

   Add yourself to the sudoers file

   

   ```` ```CODE ````

   `> sudo visudo`

   

   Add this line to the file(replace your\_username):

   

   

   ```` ```CODE ````

   `your_username ALL=(ALL) NOPASSWD:ALL`

   

   You should see docker in the output

   

   ```` ```CODE ````

   `> groups`

   `your_username sudo ... docker`

   

7. **Build Docker image and Run Jupyter**  
    Follow the instructions in the README file [here](https://github.com/gpsaggese/umd_classes/tree/master/class_project/instructions/tutorial_template/tutorial_github_data605_style).  
   **There is a typo in the path mentioned in the README, it should be *tutorial\_github\_data605\_style* and not *docker\_simple***  
     
8. **Final Step: Commands to Test Installation**  
     
   Check the installation by running:  
     
     
   ```` ```CODE ````  
   `> docker pull hello-world`  
   `Using default tag: latest`  
     
     
   Run Linter

```` ```CODE ````  
`> i lint --files="dir1/file1.py dir2/file2.py"`

Git Pull

```` ```CODE ````  
`git pull`

Creating a branch from your issue (To be done ONCE)

```` ```CODE ````  
`i git_branch_create -i issue_number`

## **Troubleshooting**

If you encounter issues:

Create a new GitHub issue in the repo.  
Assign it to @aver81 and @indro.  
Provide:  
The exact command you ran  
Your OS/platform  
Full error output (copy-paste, with screenshots)

