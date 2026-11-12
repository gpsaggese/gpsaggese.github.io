#!/bin/bash
# """
# Select GPU or CPU Docker options for the Mctx tutorial.
#
# Set MCTX_DEVICE to `auto` (default), `gpu`, or `cpu`.
# """


_has_nvidia_docker_runtime() {
    local docker_cmd
    docker_cmd=$(get_docker_cmd)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    if ! $docker_cmd info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
        return 1
    fi
    return 0
}


get_mctx_docker_run_opts() {
    local device_mode
    device_mode=${MCTX_DEVICE:-auto}
    case "$device_mode" in
        auto)
            if _has_nvidia_docker_runtime; then
                echo "MCTX_DEVICE=auto: using the NVIDIA GPU" >&2
                echo "--gpus all"
            else
                echo "MCTX_DEVICE=auto: NVIDIA runtime not found; using CPU" >&2
                echo "-e JAX_PLATFORMS=cpu"
            fi
            ;;
        gpu)
            if ! _has_nvidia_docker_runtime; then
                echo "MCTX_DEVICE=gpu but an accessible NVIDIA Docker runtime was not found" >&2
                return 1
            fi
            echo "MCTX_DEVICE=gpu: using the NVIDIA GPU" >&2
            echo "--gpus all"
            ;;
        cpu)
            echo "MCTX_DEVICE=cpu: forcing the CPU backend" >&2
            echo "-e JAX_PLATFORMS=cpu"
            ;;
        *)
            echo "Invalid MCTX_DEVICE='$device_mode'; expected auto, gpu, or cpu" >&2
            return 1
            ;;
    esac
}
