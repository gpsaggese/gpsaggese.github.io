1. What is the primary purpose of Docker Compose?
   - A) To create monolithic applications
   - B) **To manage multi-container applications on a single node**
   - C) To build operating systems
   - D) To monitor application network traffic
   - E) To develop mobile applications

2. Which line in a Docker Compose file is mandatory as the first entry?
   - A) `services:`
   - B) `networks:`
   - C) **`version:`**
   - D) `volumes:`
   - E) `containers:`

3. What does the `services` key in a Docker Compose file define?
   - A) External resources needed by the application
   - B) **Microservices that make up the application**
   - C) The hardware requirements for running the containers
   - D) The configuration for the Docker host
   - E) Workflow for data processing

4. What type of network is created by default in Docker Compose?
   - A) Host network
   - B) Overlay network
   - C) **Bridge network**
   - D) Virtual private network
   - E) Local area network

5. Which command is used to start up all services defined in a Docker Compose file?
   - A) `docker compose start`
   - B) `docker compose initialize`
   - C) **`docker compose up`**
   - D) `docker compose execute`
   - E) `docker compose activate`

6. What does the `volumes` key in a Docker Compose file allow you to do?
   - A) Define the storage type for the Docker images
   - B) **Create new volumes for data persistence**
   - C) Control the network configuration
   - D) Set up environmental variables
   - E) Manage deployment targets

7. In Docker Compose, how would you specify a custom filename for your Compose file?
   - A) **Using the `-f` option**
   - B) By renaming the file to `compose.yml`
   - C) Including it in the `version:` key
   - D) Setting it in the Docker daemon configuration
   - E) It's not possible to specify a custom filename

8. Which command would you use to view logs from running containers?
   - A) `docker compose status`
   - B) `docker compose output`
   - C) **`docker compose logs`**
   - D) `docker compose history`
   - E) `docker compose print`

9. What does the YAML key `networks:` define in a Docker Compose file?
   - A) The global settings for the Docker application
   - B) **Networks to which the services should connect**
   - C) The protocols to use for communication
   - D) Security policies for the containers
   - E) Resource allocation for the services

10. If you want to stop and remove all containers along with their networks, which command would you use?
    - A) **`docker compose down`**
    - B) `docker compose stop`
    - C) `docker compose remove`
    - D) `docker compose reset`
    - E) `docker compose terminate`

11. How is the process for building Docker containers initiated using Docker Compose?
    - A) `docker compose run`
    - B) **`docker compose build`**
    - C) `docker compose create`
    - D) `docker compose generate`
    - E) `docker compose customize`

12. What does the `top` command in Docker Compose do?
    - A) Shows the logs of the container
    - B) **Displays the running processes inside each container**
    - C) Lists the available Docker images
    - D) Initiates a network scan
    - E) Shuts down the services gracefully

13. When would you use the command `docker-compose down -v`?
    - A) To start a new service
    - B) To view container logs
    - C) **To stop services and remove associated volumes**
    - D) To update a service configuration
    - E) To check Docker service health

14. In the context of Docker Compose, what does the term "microservices" refer to?
    - A) Large standalone applications
    - B) **Small, independent services that work together**
    - C) Any service-based architecture
    - D) Server management
    - E) Database management systems

15. Which command allows you to forcefully stop all service containers?
    - A) `docker compose halt`
    - B) **`docker compose kill`**
    - C) `docker compose terminate`
    - D) `docker compose exit`
    - E) `docker compose stop -f`

16. What is a common use case for using Docker Stacks?
    - A) Manage single applications only
    - B) **Manage multi-container apps across multiple hosts**
    - C) Improve application speed
    - D) Simplify database connections
    - E) Access cloud storage

17. What would you see when executing `docker compose ps`?
    - A) The state of the network
    - B) **The list of running containers**
    - C) The storage consumption
    - D) The environmental variables
    - E) The service logs

18. In a Docker Compose file, which of the following is true about the `build:` key?
    - A) It is optional for defining network configurations
    - B) **It specifies the context for building an image**
    - C) It defines the instructions for configuring a service
    - D) It sets up the volumes for a service
    - E) It lists dependencies

19. Which Docker Compose command is used to pull service images from a registry?
    - A) **`docker compose pull`**
    - B) `docker compose fetch`
    - C) `docker compose get`
    - D) `docker compose update`
    - E) `docker compose download`

20. How do you indicate an alternate environment file in a Docker Compose command?
    - A) **Using the `--env-file` option**
    - B) Specifying it in the YAML file
    - C) Using the `-e` flag
    - D) It cannot be specified
    - E) It's set globally in Docker configuration