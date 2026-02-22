#!/bin/bash
# """
# Display versions of installed tools and packages.
#
# This script prints version information for Python, pip, Jupyter, and all
# installed Python packages. Used for debugging and documentation purposes
# to verify the Docker container environment setup.
# """

# Display Python 3 version.
echo "# Python3"
python3 --version

# Display pip version.
echo "# pip3"
pip3 --version

# Display Jupyter version.
echo "# jupyter"
jupyter --version

# List all installed Python packages and their versions.
echo "# Python packages"
pip3 list

# Template for adding additional tool versions.
# echo "# mongo"
# mongod --version
