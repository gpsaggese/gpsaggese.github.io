# Docker tutorial

## Set-up

- Make sure Docker works on your laptop
  ```
  > docker version
  Client:
   Cloud integration: v1.0.24
   Version:           20.10.17
   API version:       1.41
   Go version:        go1.17.11
   Git commit:        100c701
   Built:             Mon Jun  6 23:04:45 2022
   OS/Arch:           darwin/amd64
   Context:           default
   Experimental:      true

  Server: Docker Desktop 4.10.1 (82475)
   Engine:
    Version:          20.10.17
    API version:      1.41 (minimum version 1.12)
    Go version:       go1.17.11
    Git commit:       a89b842
    Built:            Mon Jun  6 23:01:23 2022
    OS/Arch:          linux/amd64
    Experimental:     false
   containerd:
    Version:          1.6.6
    GitCommit:        10c12954828e7c7c9b6e0ea9b0c02b01407d3ae1
   runc:
    Version:          1.1.2
    GitCommit:        v1.1.2-0-ga916309
   docker-init:
    Version:          0.19.0
    GitCommit:        de40ad0
  ```

## Images

- Pull ubuntu image, not the latest but the one we use for our tutorials

  ```
  > docker image pull ubuntu:20.04
  20.04: Pulling from library/ubuntu
  eaead16dc43b: Pull complete
  Digest: sha256:450e066588f42ebe1551f3b1a535034b6aa46cd936fe7f2c6b0d72997ec61dbd
  Status: Downloaded newer image for ubuntu:20.04
  docker.io/library/ubuntu:20.04
  ```

- Images are referred to like: `<REGISTRY_URL>/<IMAGE_REPO>:<IMAGE_TAG>`
  - E.g., `docker image pull gcr.io/google-containers/git-sync:v3.1.5`
  - If you don't specify a tag you get `latest`, which is just a special tag with
    no special meaning but just a convention to be the newest

- Now you see your local copy of the image
  ```
  > docker images
  REPOSITORY                    TAG                  IMAGE ID       CREATED        SIZE
  ubuntu                        20.04                680e5dfb52c7   2 weeks ago    72.8MB
  ```

- You can pull other images and see that they are downloaded
  ```
  > docker image pull alpine:latest
  > docker image pull redis:latest

  > docker image ls
  REPOSITORY  TAG     IMAGE ID        CREATED       SIZE
  alpine      latest  f70734b6a266    40 hours ago  5.61MB
  redis       latest  a4d3716dbb72    45 hours ago  98.3MB
  ```
- It's very simple to install an entire OS or a tool

- Note that the same image can have multiple tags
  ```
  > docker image pull ubuntu
  Using default tag: latest
  latest: Pulling from library/ubuntu
  Digest: sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2
  Status: Image is up to date for ubuntu:latest
  docker.io/library/ubuntu:latest

  > docker image pull ubuntu:22.04
  22.04: Pulling from library/ubuntu
  Digest: sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2
  Status: Downloaded newer image for ubuntu:22.04
  docker.io/library/ubuntu:22.04

  > docker image pull ubuntu:latest
  latest: Pulling from library/ubuntu
  Digest: sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2
  Status: Image is up to date for ubuntu:latest
  docker.io/library/ubuntu:latest

  > docker images | grep ubuntu
  ubuntu              22.04                a8780b506fa4   5 days ago     77.8MB
  ubuntu              latest               a8780b506fa4   5 days ago     77.8MB
  ```

- Dangling images
  ```
  > docker image ls | grep none
  REPOSITORY     TAG                   IMAGE ID        CREATED     SIZE
  <none>      <none>               16bc6726a51c   41 hours ago   2.33GB
  <none>      <none>               cd7908d486d5   2 days ago     1.72GB
  ```

- You can see the layers of the image with:
  ```
  > docker image inspect ubuntu:latest
  [
      {
          "Id": "sha256:a8780b506fa4eeb1d0779a3c92c8d5d3e6a656c758135f62826768da458b5235",
          "RepoTags": [
              "ubuntu:22.04",
              "ubuntu:latest"
          ],
          "RepoDigests": [
              "ubuntu@sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2"
          ],
  ...
  ```

- You can see how each layer was created with:
  ```
  > docker image history ubuntu:20.04
  IMAGE          CREATED       CREATED BY                                      SIZE      COMMENT
  680e5dfb52c7   2 weeks ago   /bin/sh -c #(nop)  CMD ["bash"]                 0B
  <missing>      2 weeks ago   /bin/sh -c #(nop) ADD file:7633003155a105941…   72.8MB
  ```

- Deleting an image removes all the layers from the Docker host, unless those
  layers are needed by other images

  ```bash
  > docker image rm
  ```

- To delete all the images:

  ```bash
  > docker image rm -f $(docker image ls -q)
  ```

## Containers

- Start a simple container and run a bash shell inside
  ```bash
  > docker run -it ubuntu:20.04 /bin/bash
  root@cc0d91a7ac4e:/#
  ```
- A container exits when the main process exits

- Changes to the container are persisted until the container is not killed

  - You can do an experiment

- Start, stop, restart a container

  ```
  # Start.
  > docker container run --name percy -it ubuntu:latest /bin/bash
  # Stop.
  > docker container stop percy
  # The container shows "Exited (0)" since it is stopped.
  > docker container ls -a
  CNTNR ID  IMAGE         COMMAND   CREATED   STATUS      NAMES
  9cb...65  ubuntu:latest /bin/bash 4mins     Exited(0)   percy
  # Restart.
  > docker container start percy
  > docker container ls
  CONTAINER ID  IMAGE          COMMAND      CREATED  STATUS     NAMES
  9cb2d2fd1d65  ubuntu:latest  "/bin/bash"  4 mins   Up 3 secs  percy
  # Log in from a different window.
  > docker container exec -it percy bash
  # Kill the container.
  > docker container rm -f percy
  ```

- Find the running containers

  ```bash
  > docker container ls
  ```

## Building containers

  ```
  > echo "flask" >requirements.txt
  ```

- Example Dockerfile
  ```
  FROM python:3.8-slim-buster
  LABEL maintainer="gsaggese@umd.edu"

  WORKDIR /app

  COPY requirements.txt requirements.txt
  RUN pip3 install -r requirements.txt

  COPY . .

  CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
  ```

- Then you can build the image
  ```
  > docker build -t test . --progress plain
  #1 [internal] load build definition from Dockerfile
  #1 sha256:d4ee5c5c88ac988faa4fa4b57712c90dd1c6d0cdbd63f8f3eab15568437a2139
  #1 transferring dockerfile: 72B done
  #1 DONE 0.0s

  #2 [internal] load .dockerignore
  #2 sha256:2514d22547df98b6fab8b1c2d6fdb85e46ee0f83962624b98edd74719507b633
  #2 transferring context: 2B done
  #2 DONE 0.0s

  #3 [internal] load metadata for docker.io/library/python:3.8-slim-buster
  #3 sha256:a82ddfb0a3c3ab3f4e2ebc7582cec39f26df7d1ae41d54f70ea9fe596d7b25c7
  #3 DONE 0.4s

  #6 [internal] load build context
  #6 sha256:632017a5347da188dd3ef32e87156a2cbf588f3ed3759c0ce0a32636fbc71bc9
  #6 transferring context: 157B done
  #6 DONE 0.0s

  #4 [1/5] FROM docker.io/library/python:3.8-slim-buster@sha256:4cda66a01b5571bd4f3d634b301f72e580a94b2c1ce87ead057040a6dea4a416
  #4 sha256:a61210c0cc9cadb082453dd5d7c25f2218f7f3a0b87e9f60390709f702d6097d
  #4 resolve docker.io/library/python:3.8-slim-buster@sha256:4cda66a01b5571bd4f3d634b301f72e580a94b2c1ce87ead057040a6dea4a416 0.0s done
  #4 sha256:4cda66a01b5571bd4f3d634b301f72e580a94b2c1ce87ead057040a6dea4a416 988B / 988B done
  #4 sha256:e659e823e3fb173b0ff6a7e905042c18cc4eb117650f3ba74f2b69239cc8d4cc 1.37kB / 1.37kB done
  #4 sha256:6ba145ad2ad6bd0be9290f841bdb5726ccfaf54a72bb3a25ea104ec031ee4655 7.53kB / 7.53kB done
  ...
  #4 DONE 4.6s

  #5 [2/5] WORKDIR /app
  #5 sha256:a7f97bcc0c3e4c1cbe3958f426cfb8f7b4b399ddd456f9589d41afeb9596d59a
  #5 DONE 0.3s

  #7 [3/5] COPY requirements.txt requirements.txt
  #7 sha256:ca021573e8b1a9f6c302b924d680666d15b7b6e299bcc776208b70ca1798f902
  #7 DONE 0.0s

  #8 [4/5] RUN pip3 install -r requirements.txt
  #8 sha256:5847826674ef61e0d911e93cbfdce62db83c718ac6dca0e8aaaf07cff41331a1
  #8 2.563 Collecting flask
  #8 2.611   Downloading Flask-2.2.2-py3-none-any.whl (101 kB)
  #8 2.628      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.5/101.5 KB 7.4 MB/s eta 0:00:00
  #8 2.683 Collecting Jinja2>=3.0
  #8 2.692   Downloading Jinja2-3.1.2-py3-none-any.whl (133 kB)
  ...
  #8 3.662 Installing collected packages: zipp, MarkupSafe, itsdangerous, click, Werkzeug, Jinja2, importlib-metadata, flask
  #8 4.311 Successfully installed Jinja2-3.1.2 MarkupSafe-2.1.1 Werkzeug-2.2.2 click-8.1.3 flask-2.2.2 importlib-metadata-6.0.0 itsdangerous-2.1.2 zipp-3.11.0
  #8 4.311 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
  #8 4.440 WARNING: You are using pip version 22.0.4; however, version 22.3.1 is available.
  #8 4.440 You should consider upgrading via the '/usr/local/bin/python -m pip install --upgrade pip' command.
  #8 DONE 4.6s

  #9 [5/5] COPY . .
  #9 sha256:09234d169450c171b97ec72395869aa54852616d8d9a28d23e94397d731d4653
  #9 DONE 0.0s

  #10 exporting to image
  #10 sha256:e8c613e07b0b7ff33893b694f7759a10d42e180f2b4dc349fb57dc6b71dcab00
  #10 exporting layers
  #10 exporting layers 0.2s done
  #10 writing image sha256:cc5a300674334a27a40e1306883bc9330d7b40190490f00b3464572eb9bd3463 done
  #10 naming to docker.io/library/test done
  #10 DONE 0.2s
  ```

## Docker Basics

- A docker container is an instance of a _container image_ that runs on your
  machine and is managed by Docker
  - A large variety of docker images have been uploaded to
    [DockerHub](https://hub.docker.com/), that have been created by both the
    Docker developers and the Docker user community.

- Each docker user has a certain set of container images stored in his/her
  machine. Those images are managed by the Docker platform itself. To see the
  images that are stored in your machine:
  ```
  > docker images
  ```
- Every image is associated with a unique `repository-name`, which can be
  thought of as a readable unique image identifier

### Managing Containers

- A basic way to _start_ a docker container is:
  ```
  > docker run --name my-container -d repository-name
  ```

- The above command starts a container based on the image `repository-name`. If
  the specific image is _not_ present on your machine, docker automatically
  `pulls` the image from DockerHub

- The parameters:
  - `--name`: Allows you to specify a name for your container; if you don't add
    this parameter, docker randomly generates a name for you
  - `-d`: Runs the container in the background (stands for "detached")

- Once you start a container in the background, it will keep running until you
  `stop` it like so:

  ```
  > docker stop my-container
  ```

- You can `restart` your container by simply running:

  ```
  > docker restart my-container
  ```

- If you are done with a container and want to completely remove it from your
  system (remember that containers take up non-trivial amounts of disk space!),
  after `stopping` the container you can run:
  ```
  > docker rm my-container
  ```

- To see the docker containers that are currently running:
  ```
  > docker ps
  ```

- Adding a `-a` flag shows you both the running and stopped containers that are
  instantiated on your machine.

- If something goes wrong, for example the files or data within the container get
  corrupted and you want to re-initialize the container, you need to `stop`
  the container, `remove` it, and then re-initialize it from scratch.

  ```
  > docker stop my-container
  > docker rm my-container
  > docker run --name my-container -d repository-name
  ```

### Accessing Containers

- A container is very similar to a virtual machine, viewed at a high level. That
  means it resembles an independent environment within your machine environment.
  One good way to run programs within your container is to connect to the
  container's `bash` shell and run your programs from within the container
  itself:

  ```
  > docker exec -it my-container bash
  ```

  - `-it`: Instructs Docker to allocate a pseudo-TTY connected to the container’s
    `stdin` (standard input); creating an interactive bash shell in the container.

### Copying Files into Containers

Since containers have their own independent environment and file system, they
can only see the files within that file system. In some situations you will want
to write code in your host machine\'s environment and then run that piece of
code within the container. To copy files into the container you can:

```
> docker cp file container-name:/path/where/to/copy/file
```

### Docker Help

- To find out more about the different docker commands:
  ```
  > docker COMMAND --help
  ```
