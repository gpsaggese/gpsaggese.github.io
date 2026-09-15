#!/bin/bash
# """
# Utility functions for Docker container management.
# """


# #############################################################################
# General utilities
# #############################################################################


run() {
    # """
    # Execute a command with echo output.
    #
    # :param cmd: Command string to execute
    # :return: Exit status of the executed command
    # """
    cmd="$*"
    echo "> $cmd"
    eval "$cmd"
}


enable_verbose_mode() {
    # """
    # Enable shell command tracing (set -x) when VERBOSE is set to 1.
    #
    # Reads the VERBOSE variable set by parse_docker_jupyter_args.
    # Call this after parsing args to activate tracing for the rest of the script.
    # """
    if [[ $VERBOSE == 1 ]]; then
        set -x
    fi
}


# #############################################################################
# Argument parsing
# #############################################################################


_print_default_help() {
    # """
    # Print usage information and available default options for docker scripts.
    # """
    echo "Usage: $(basename $0) [options]"
    echo ""
    echo "Options:"
    echo "  -f    Force kill existing container with same name before starting"
    echo "  -h    Print this help message and exit"
    echo "  -v    Enable verbose output (set -x)"
}


parse_default_args() {
    # """
    # Parse default command-line arguments for docker scripts.
    #
    # Sets VERBOSE and FORCE variables in the caller's scope. Enables set -x
    # when -v is passed. Prints help and exits when -h is passed.
    # Updates OPTIND so the caller can shift away processed arguments.
    #
    # :param @: command-line arguments forwarded from the calling script
    # """
    VERBOSE=0
    FORCE=0
    while getopts "fhv" flag; do
        case "${flag}" in
            f) FORCE=1;;
            h) _print_default_help; exit 0;;
            v) VERBOSE=1;;
            *) _print_default_help; exit 1;;
        esac
    done
    enable_verbose_mode
}


_print_docker_jupyter_help() {
    # """
    # Print usage information and available options for docker_jupyter.sh.
    # """
    echo "Usage: $(basename $0) [options]"
    echo ""
    echo "Launch Jupyter Lab inside a Docker container."
    echo ""
    echo "Options:"
    echo "  -f          Force kill existing container with same name before starting"
    echo "  -h          Print this help message and exit"
    echo "  -p PORT     Host port to forward to Jupyter Lab (default: 8888)"
    echo "  -u          Enable vim keybindings in Jupyter Lab"
    echo "  -v          Enable verbose output (set -x)"
}


parse_docker_jupyter_args() {
    # """
    # Parse command-line arguments for docker_jupyter.sh.
    #
    # Sets JUPYTER_HOST_PORT, JUPYTER_USE_VIM, TARGET_DIR, VERBOSE, FORCE, and
    # OLD_CMD_OPTS in the caller's scope.  Enables set -x when -v is passed.
    # Prints help and exits when -h is passed.
    #
    # :param @: command-line arguments forwarded from the calling script
    # """
    # Set defaults.
    JUPYTER_HOST_PORT=8888
    JUPYTER_USE_VIM=0
    VERBOSE=0
    FORCE=0
    # Save original args to pass through to run_jupyter.sh.
    OLD_CMD_OPTS="$*"
    # Parse options.
    while getopts "fhp:uv" flag; do
        case "${flag}" in
            f) FORCE=1;;
            h) _print_docker_jupyter_help; exit 0;;
            p) JUPYTER_HOST_PORT=${OPTARG};;  # Port for Jupyter Lab.
            u) JUPYTER_USE_VIM=1;;            # Enable vim bindings.
            v) VERBOSE=1;;                    # Enable verbose output.
            *) _print_docker_jupyter_help; exit 1;;
        esac
    done
    # Enable command tracing if verbose mode is requested.
    enable_verbose_mode
}


# #############################################################################
# Docker engine detection
# #############################################################################


get_docker_engine() {
    # """
    # Return the container engine to use ("docker" or "apple").
    #
    # Priority:
    # 1. DOCKER_ENGINE env var if set.
    # 2. Auto-detect: "apple" only on gp's macOS laptop (Darwin + hostname
    #    "gpmac.local"), "docker" otherwise.
    # """
    if [[ -n "$DOCKER_ENGINE" ]]; then
        # Use the variable.
        echo "$DOCKER_ENGINE"
    elif [[ "$(uname -s)" == "Darwin" && "$(uname -n)" == "gpmac.local" ]]; then
        # Only on GP's laptop use Apple containers.
        echo "apple"
    else
        echo "docker"
    fi
}


get_docker_cmd() {
    # """
    # Return the CLI command string for the active engine.
    #
    # :return: "docker" for the docker engine, "container" for Apple engine
    # """
    local engine
    engine=$(get_docker_engine)
    if [[ "$engine" == "apple" ]]; then
        echo "container"
    else
        echo "docker"
    fi
}


# #############################################################################
# Docker image management
# #############################################################################


get_docker_vars_script() {
    # """
    # Load Docker variables from docker_name.sh script.
    #
    # :param script_path: Path to the script to determine the Docker configuration directory
    # :return: Sources REPO_NAME, IMAGE_NAME, and FULL_IMAGE_NAME variables
    # """
    local script_path=$1
    # Find the name of the container.
    SCRIPT_DIR=$(dirname $script_path)
    DOCKER_NAME="$SCRIPT_DIR/docker_name.sh"
    if [[ ! -e $SCRIPT_DIR ]]; then
        echo "Can't find $DOCKER_NAME"
        exit -1
    fi;
    source $DOCKER_NAME
}


print_docker_vars() {
    # """
    # Print current Docker variables to stdout.
    # """
    echo "# Docker Vars"
    echo "REPO_NAME=$REPO_NAME"
    echo "IMAGE_NAME=$IMAGE_NAME"
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    echo "DOCKER_ENGINE=$(get_docker_engine)"
    echo "DOCKER_CMD=$(get_docker_cmd)"
}


build_container_image() {
    # """
    # Build a Docker container image.
    #
    # Supports both single-architecture and multi-architecture builds.
    # Creates temporary build directory, copies files, and builds the image.
    #
    # :param @: Additional options to pass to docker build/buildx build
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    # Prepare build area.
    #tar -czh . | $DOCKER_CMD build $OPTS -t $IMAGE_NAME -
    DIR="../tmp.build"
    if [[ -d $DIR ]]; then
        rm -rf $DIR
    fi;
    cp -Lr . $DIR || true
    # Build container.
    echo "DOCKER_BUILDKIT=$DOCKER_BUILDKIT"
    echo "DOCKER_BUILD_MULTI_ARCH=$DOCKER_BUILD_MULTI_ARCH"
    if [[ $DOCKER_BUILD_MULTI_ARCH != 1 ]]; then
        # Build for a single architecture.
        echo "Building for current architecture..."
        OPTS="--progress plain $@"
        (cd $DIR; $DOCKER_CMD build $OPTS -t $FULL_IMAGE_NAME . 2>&1 | tee ../docker_build.log; exit ${PIPESTATUS[0]})
    else
        # Multi-arch builds require docker buildx; force docker CLI.
        echo "Building for multiple architectures..."
        OPTS="$@"
        export DOCKER_CLI_EXPERIMENTAL=enabled
        # Create a new builder.
        #docker buildx rm --all-inactive --force
        #docker buildx create --name mybuilder
        #docker buildx use mybuilder
        # Use the default builder.
        docker buildx use multiarch
        docker buildx inspect --bootstrap
        # Note that one needs to push to the repo since otherwise it is not
        # possible to keep multiple.
        (cd $DIR; docker buildx build --push --platform linux/arm64,linux/amd64 $OPTS --tag $FULL_IMAGE_NAME . 2>&1 | tee ../docker_build.log; exit ${PIPESTATUS[0]})
        # Report the status.
        docker buildx imagetools inspect $FULL_IMAGE_NAME
    fi;
    # Report build version.
    if [ -f docker_build.version.log ]; then
      rm docker_build.version.log
    fi
    (cd $DIR; $DOCKER_CMD run --rm -it -v $(pwd):/data $FULL_IMAGE_NAME bash -c "/data/version.sh") 2>&1 | tee docker_build.version.log
    #
    $DOCKER_CMD image ls | grep "$REPO_NAME/$IMAGE_NAME" || true
    rm -rf $DIR
    echo "*****************************"
    echo "SUCCESS"
    echo "*****************************"
}


remove_container_image() {
    # """
    # Remove Docker container image(s) matching the current configuration.
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    $DOCKER_CMD image ls | grep $FULL_IMAGE_NAME
    $DOCKER_CMD image ls | grep $FULL_IMAGE_NAME | awk '{print $1}' | xargs -n 1 -t $DOCKER_CMD image rm -f
    $DOCKER_CMD image ls
    echo "${FUNCNAME[0]} ... done"
}


push_container_image() {
    # """
    # Push Docker container image to registry.
    #
    # Authenticates using credentials from ~/.docker/passwd.$REPO_NAME.txt.
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    $DOCKER_CMD login --username $REPO_NAME --password-stdin <~/.docker/passwd.$REPO_NAME.txt
    $DOCKER_CMD images $FULL_IMAGE_NAME
    $DOCKER_CMD push $FULL_IMAGE_NAME
    echo "${FUNCNAME[0]} ... done"
}


pull_container_image() {
    # """
    # Pull Docker container image from registry.
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    $DOCKER_CMD pull $FULL_IMAGE_NAME
    echo "${FUNCNAME[0]} ... done"
}


# #############################################################################
# Docker container management
# #############################################################################


kill_container() {
    # """
    # Kill and remove Docker container(s) matching the current configuration.
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    $DOCKER_CMD container ls
    #
    CONTAINER_ID=$($DOCKER_CMD container ls -a | grep $FULL_IMAGE_NAME | awk '{print $1}')
    echo "CONTAINER_ID=$CONTAINER_ID"
    if [[ ! -z $CONTAINER_ID ]]; then
        $DOCKER_CMD container rm -f $CONTAINER_ID
        $DOCKER_CMD container ls
    fi;
    echo "${FUNCNAME[0]} ... done"
}


kill_container_by_name() {
    # """
    # Kill and remove a Docker container by its name.
    #
    # :param container_name: Name of the container to kill
    # """
    local container_name=$1
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    echo "# ${FUNCNAME[0]}: $container_name"
    # Apple's container list doesn't support --filter, so we try rm -f
    # directly and ignore errors if the container doesn't exist.
    local rm_opts
    if [[ "$(get_docker_engine)" == "apple" ]]; then
        rm_opts=("$DOCKER_CMD" "rm" "-f")
    else
        rm_opts=("$DOCKER_CMD" "container" "rm" "-f")
    fi
    # Try to find and kill any container matching this name.
    CONTAINER_ID=$("${rm_opts[@]}" "$container_name" 2>/dev/null) || true
    echo "Container '$container_name' cleaned up"
    echo "${FUNCNAME[0]} ... done"
}


exec_container() {
    # """
    # Execute bash shell in running Docker container.
    #
    # Opens an interactive bash session in the first container matching the
    # current configuration.
    # """
    echo "# ${FUNCNAME[0]} ..."
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
    echo "FULL_IMAGE_NAME=$FULL_IMAGE_NAME"
    $DOCKER_CMD container ls
    #
    CONTAINER_ID=$($DOCKER_CMD container ls -a | grep $FULL_IMAGE_NAME | awk '{print $1}')
    echo "CONTAINER_ID=$CONTAINER_ID"
    $DOCKER_CMD exec -it $CONTAINER_ID bash
    echo "${FUNCNAME[0]} ... done"
}


# #############################################################################
# Docker common options
# #############################################################################


get_docker_common_options() {
    # """
    # Return docker run options common to all container types.
    #
    # Includes volume mount for the git root, plus environment variables for
    # PYTHONPATH and host OS name.
    #
    # :return: docker run options string with volume mounts and env vars
    # """
    echo "-v $GIT_ROOT:/git_root \
    -e PYTHONPATH=/git_root:/git_root/helpers_root:/git_root/msml610/tutorials \
    -e CSFY_GIT_ROOT_PATH=/git_root \
    -e CSFY_HOST_OS_NAME=$(uname -s) \
    -e CSFY_HOST_NAME=$(uname -n) \
    -e DOCKER_ENGINE=$(get_docker_engine)"
}


# #############################################################################
# Docker bash
# #############################################################################


get_docker_bash_command() {
    # """
    # Return the base docker run command for an interactive bash shell.
    #
    # :return: docker run command string with --rm and -ti flags
    # """
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    if [ -t 0 ]; then
        echo "$DOCKER_CMD run --rm -ti"
    else
        echo "$DOCKER_CMD run --rm -i"
    fi
}


get_docker_bash_options() {
    # """
    # Return docker run options for a Docker container.
    #
    # :param container_name: Name for the Docker container
    # :param port: Port number to forward (optional, skipped if empty)
    # :param extra_opts: Additional docker run options (optional)
    # :return: docker run options string with name, volume mounts, and env vars
    # """
    local container_name=$1
    local port=$2
    local extra_opts=$3
    local port_opt=""
    if [[ -n $port ]]; then
        port_opt="-p $port:$port"
    fi
    echo "--name $container_name \
    $port_opt \
    $extra_opts \
    $(get_docker_common_options)"
}


# #############################################################################
# Docker cmd
# #############################################################################


get_docker_cmd_command() {
    # """
    # Return the base docker run command for executing a non-interactive command.
    #
    # :return: docker run command string with --rm and -i flags
    # """
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    echo "$DOCKER_CMD run --rm -i"
}


# #############################################################################
# Docker Jupyter
# #############################################################################


get_docker_jupyter_command() {
    # """
    # Return the base docker run command for running Jupyter Lab interactively.
    #
    # :return: docker run command string with --rm and -ti flags (if TTY available)
    # """
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    local cmd="$DOCKER_CMD run --rm"
    # Add interactive and TTY flags only if stdin is a TTY.
    if [[ -t 0 ]]; then
        cmd="$cmd -ti"
    fi
    echo "$cmd"
}


get_docker_jupyter_options() {
    # """
    # Return docker run options for a Jupyter Lab container.
    #
    # :param container_name: Name for the Docker container
    # :param host_port: Host port to forward to container port 8888
    # :param jupyter_use_vim: 0 or 1 to enable vim bindings
    # :return: docker run options string
    # """
    local container_name=$1
    local host_port=$2
    local jupyter_use_vim=$3
    # Run as the current user when user is saggese.
    if [[ "$(whoami)" == "saggese" ]]; then
        echo "Overwriting jupyter_use_vim since user='saggese'" >&2
        jupyter_use_vim=1
    fi
    local port_opt=""
    # Apple container engine's -p port forwarding is broken (v1.0.0).
    # We skip it and use an external port forwarder via bridge100 instead.
    if [[ "$(get_docker_engine)" != "apple" ]]; then
        port_opt="-p $host_port:8888"
    fi
    echo "--name $container_name \
    $port_opt \
    $(get_docker_common_options) \
    -e JUPYTER_USE_VIM=$jupyter_use_vim"
}


get_container_ip() {
    # """
    # Get the bridge100 IP of a running Apple container.
    #
    # Apple containers get a 192.168.64.x IP on the vmnet bridge100 interface.
    # Port forwarding via -p is broken, so we need this IP for direct access.
    #
    # :param container_name: Name of the running container
    # :return: Container IP on bridge100
    # """
    local container_name=$1
    local docker_cmd
    docker_cmd=$(get_docker_cmd)
    local ip
    ip=$($docker_cmd exec "$container_name" hostname -I 2>/dev/null | awk '{print $1}')
    echo "$ip"
}


configure_jupyter_vim_keybindings() {
    # """
    # Configure JupyterLab vim keybindings based on JUPYTER_USE_VIM env var.
    #
    # Reads JUPYTER_USE_VIM; if 1, verifies jupyterlab_vim is installed and
    # writes enabled settings; otherwise writes disabled settings.
    # """
    mkdir -p ~/.jupyter/lab/user-settings/@axlair/jupyterlab_vim
    if [[ $JUPYTER_USE_VIM == 1 ]]; then
        # Check that jupyterlab_vim is installed before trying to enable it.
        if ! pip show jupyterlab_vim > /dev/null 2>&1; then
            echo "ERROR: jupyterlab_vim is not installed but vim bindings were requested."
            echo "Install it with: pip install jupyterlab_vim"
            exit 1
        fi
        echo "Enabling vim."
        cat <<EOF > ~/.jupyter/lab/user-settings/\@axlair/jupyterlab_vim/plugin.jupyterlab-settings
{
    "enabled": true,
    "enabledInEditors": true,
    "extraKeybindings": [],
}
EOF
    else
        echo "Disabling vim."
        cat <<EOF > ~/.jupyter/lab/user-settings/\@axlair/jupyterlab_vim/plugin.jupyterlab-settings
{
    "enabled": false,
    "enabledInEditors": false,
    "extraKeybindings": [],
}
EOF
    fi;
}


configure_jupyter_notifications() {
    # """
    # Disable JupyterLab news fetching and update checks.
    # """
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
    cat <<EOF > ~/.jupyter/lab/user-settings/\@jupyterlab/apputils-extension/notification.jupyterlab-settings
{
    // Notifications
    // @jupyterlab/apputils-extension:notification
    // Notifications settings.

    // Fetch official Jupyter news
    // Whether to fetch news from the Jupyter news feed. If Always (`true`), it will make a request to a website.
    "fetchNews": "false",
    "checkForUpdates": false
}
EOF
}


configure_jupyter_autosave() {
    # """
    # Configure JupyterLab global autosave interval to 6 seconds.
    # """
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/docmanager-extension
    cat <<EOF > ~/.jupyter/lab/user-settings/\@jupyterlab/docmanager-extension/plugin.jupyterlab-settings
{
    "autosaveInterval": 6
}
EOF
}


check_jupytext_installed() {
    # """
    # Verify that jupytext is installed before starting Jupyter Lab.
    #
    # Jupytext is required for pair notebook/Python file functionality.
    # Exits with error if jupytext is not installed.
    # """
    if ! pip show jupytext > /dev/null 2>&1; then
        echo "ERROR: jupytext is not installed but is required to run Jupyter Lab."
        echo "Install it with: pip install jupytext"
        exit 1
    fi
}


setup_jupyter_environment() {
    # """
    # Configure Jupyter Lab environment before launching.
    #
    # Performs all necessary setup steps:
    # - Configure vim keybindings
    # - Disable notifications
    # - Configure autosave interval
    # - Verify jupytext is installed
    # """
    configure_jupyter_vim_keybindings
    configure_jupyter_notifications
    configure_jupyter_autosave
    check_jupytext_installed
}


get_jupyter_args() {
    # """
    # Print the standard Jupyter Lab command-line arguments.
    #
    # :return: space-separated Jupyter Lab args for port 8888 with no browser,
    #   allow root, and no authentication
    # """
    echo "--port=8888 --no-browser --ip=0.0.0.0 --allow-root --ServerApp.token='' --ServerApp.password=''"
}


get_run_jupyter_cmd() {
    # """
    # Return the command to run run_jupyter.sh inside a container.
    #
    # Computes the script's path relative to GIT_ROOT and builds the
    # corresponding /git_root/... path used inside the container.
    #
    # :param script_path: path of the calling script (pass ${BASH_SOURCE[0]})
    # :param cmd_opts: options to forward to run_jupyter.sh
    # :return: full command string to run run_jupyter.sh
    # """
    local script_path=$1
    local cmd_opts=$2
    local script_dir
    script_dir=$(cd "$(dirname "$script_path")" && pwd)
    local rel_dir="${script_dir#${GIT_ROOT}/}"
    echo "/git_root/${rel_dir}/run_jupyter.sh $cmd_opts"
}


list_and_inspect_docker_image() {
    # """
    # List available Docker images and inspect their architecture.
    #
    # Lists all images matching FULL_IMAGE_NAME and attempts to inspect
    # their architecture using docker manifest inspect.
    #
    # Skipped for the Apple engine: `container manifest inspect` needs the
    # `container-manifest` plugin, which is not bundled with the `container`
    # CLI, so it always fails with "Plugin 'container-manifest' not found".
    # """
    local DOCKER_CMD
    DOCKER_CMD=$(get_docker_cmd)
    run "$DOCKER_CMD image ls | grep '$FULL_IMAGE_NAME' || true"
    if [[ "$(get_docker_engine)" != "apple" ]]; then
        ($DOCKER_CMD manifest inspect $FULL_IMAGE_NAME | grep arch) || true
    fi
}


kill_existing_container_if_forced() {
    # """
    # Kill existing container if FORCE flag is set.
    #
    # If FORCE is set to 1, kills and removes the container with name
    # CONTAINER_NAME. This is typically set by the -f flag.
    # """
    if [[ $FORCE == 1 ]]; then
        kill_container_by_name $CONTAINER_NAME
    fi
}
