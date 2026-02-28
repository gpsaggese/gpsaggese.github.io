"""
Import as:

import tutorials.Autogen.tests.docker_build_test as tatdbute
"""

import subprocess

# Docker image name
DOCKER_IMAGE = "gpsaggese/umd_autogen:latest"


def build_container() -> bool:
    """
    Build the Docker container.

    Returns
    -------
    bool
        True if the container builds successfully.
    """
    print("Building Docker container...")
    result = subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "."],
        capture_output=True,
        text=True,
        check=False,  # We'll handle errors manually
    )
    if result.returncode != 0:
        print("Docker build failed:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Docker build failed")
    print("Docker container built successfully.")
    return True


def test_container_builds() -> None:
    """
    Integration test to ensure Docker container builds successfully.
    """
    assert build_container()


if __name__ == "__main__":
    build_container()
    print(
        f"\nYou can now run the container interactively with:\n"
        f"docker run --rm -it -v $(pwd):/curr_dir {DOCKER_IMAGE} /bin/bash"
    )
