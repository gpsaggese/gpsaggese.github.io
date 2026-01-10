1. What is the primary purpose of containerizing an application?
   - A) To improve application installation speed
   - B) To enhance operating system performance
   - C) **To create a container with the app and its dependencies**
   - D) To reduce application code size
   - E) To enhance application design

2. In a Dockerfile, what is the function of the `COPY` command?
   - A) To install system packages
   - B) **To copy files and directories into the container**
   - C) To run a command inside a container
   - D) To expose container ports
   - E) To define environment variables

3. Which command is used to build a Docker image?
   - A) `docker run`
   - B) `docker create`
   - C) **`docker image build`**
   - D) `docker start`
   - E) `docker push`

4. What does the `WORKDIR` instruction do in a Dockerfile?
   - A) Sets the image version
   - B) **Sets the working directory for commands that follow**
   - C) Defines the network configuration
   - D) Installs required packages
   - E) Specifies the entry point of the application

5. What command would you use to display all available Docker images?
   - A) `docker ps`
   - B) `docker containers`
   - C) **`docker images`**
   - D) `docker list`
   - E) `docker show`

6. What is a build context in Docker?
   - A) The name of the Dockerfile
   - B) **The set of files sent to the Docker engine to build an image**
   - C) The command used to run a container
   - D) The environment where the container runs
   - E) The version of Docker being used

7. In the example Dockerfile provided, what is the purpose of the `RUN` command?
   - A) To copy files into the container
   - B) **To execute a command inside the container, such as installing packages**
   - C) To set environment variables
   - D) To define the entry point
   - E) To expose container ports

8. What does the command `docker container ls` display?
   - A) A list of all images
   - B) **A list of running containers**
   - C) Information about Docker networks
   - D) All volumes in Docker
   - E) The Docker version

9. Which of the following is the correct command to delete an image?
   - A) `docker erase`
   - B) **`docker rmi`**
   - C) `docker remove`
   - D) `docker delete`
   - E) `docker stop`

10. In the context of Docker, what does the term "volume" refer to?
    - A) A specific command to run
    - B) **A persistent storage mechanism for containers**
    - C) The network configuration of an image
    - D) A method to bundle application dependencies
    - E) The size of the Docker image

11. If you want to remove all stopped containers, which command should you execute?
    - A) `docker rmi $(docker images -q)`
    - B) **`docker container rm $(docker container ls -q)`**
    - C) `docker prune -a`
    - D) `docker image rm $(docker images -q)`
    - E) `docker volume rm $(docker volume ls -q)`

12. Why is it recommended to place the Dockerfile in the root of the build context?
    - A) It reduces the size of the image
    - B) **It allows the Docker build process to access all necessary files easily**
    - C) It improves the running speed of the container
    - D) It restricts the build context to only what's inside the Dockerfile
    - E) It is a requirement of Docker

13. What do the `EXPOSE` command in a Dockerfile signify?
    - A) **It indicates which ports the container listens on during runtime**
    - B) It specifies environment variables
    - C) It installs dependencies
    - D) It defines the container's command to run
    - E) It changes the working directory

14. When you push a Docker image to a registry, what must occur first?
    - A) **The image must be built successfully**
    - B) The image must be deleted
    - C) The container must be running
    - D) The Dockerfile must be updated
    - E) You must create a new volume

15. Which of the following commands would you use to show all networks in Docker?
    - A) `docker container ls`
    - B) **`docker network ls`**
    - C) `docker volumes`
    - D) `docker image ls`
    - E) `docker status`

16. After running `docker run`, what does the command typically execute?
    - A) **Starts a new container from an image**
    - B) Deletes an existing container
    - C) Builds a new image
    - D) Shows the status of images
    - E) Lists all volumes

17. When defining the `CMD` in a Dockerfile, what is its purpose?
    - A) **Specifies the default command to run when a container starts**
    - B) Sets the working directory
    - C) Installs packages
    - D) Defines environment variables
    - E) Runs build context commands

18. In the Docker tutorial, what is a primary objective of using Docker?
    - A) To increase the complexity of deployment
    - B) **To simplify the distribution and running of applications**
    - C) To create virtual machines
    - D) To separate databases from applications
    - E) To enhance security features

19. What command would you use to halt the Docker daemon?
    - A) `docker stop`
    - B) `docker shutdown`
    - C) **`systemctl stop docker`** 
    - D) `docker halt`
    - E) `docker daemon stop`

20. Which of the following is NOT a benefit of using Docker?
    - A) Easy application distribution
    - B) Isolation of application dependencies
    - C) **Increased system resource usage**
    - D) Consistency across development environments
    - E) Simplified version control of applications