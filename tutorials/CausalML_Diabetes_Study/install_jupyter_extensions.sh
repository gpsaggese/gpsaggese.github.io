#!/usr/bin/env bash
# """
# Install and configure Jupyter Notebook extensions.
#
# This script sets up Jupyter with a curated set of extensions for enhanced
# notebook functionality including code prettification, table of contents,
# execution time tracking, vim bindings, and more. It also configures the
# Jupyter data directory and extension settings.
# """

# Exit on error and print commands.
set -ex

echo "# Install Jupyter extensions"

# Create Jupyter data directory if it doesn't exist.
DIR_NAME=$(jupyter --data-dir)
echo "Jupyter data dir: $DIR_NAME"
if [[ ! -d $DIR_NAME ]]; then
  mkdir -p $DIR_NAME
fi;

# Install nbextensions package.
jupyter contrib nbextension install

# Define list of extensions to enable.
extensions="
autosavetime/main
code_prettify/code_prettify
collapsible_headings/main
comment-uncomment/main
contrib_nbextensions_help_item/main
execute_time/ExecuteTime
highlighter/highlighter
jupyter-js-widgets/extension
notify/notify
runtools/main
toc2/main
spellchecker/main"

# Enable each extension in the list.
for v in $extensions; do
  jupyter nbextension enable $v
done;

# Configure Jupyter to show incompatible extensions.
echo "{\"nbext_hide_incompat\": false}" > /$HOME/.jupyter/nbconfig/common.json

# Create nbextensions directory if it doesn't exist.
DIR=$(jupyter --data-dir)/nbextensions
if [[ ! -e $DIR ]]; then
    mkdir $DIR
fi

# Install vim bindings extension from GitHub repository.
cd $DIR
if [[ -e vim_binding ]]; then
    rm -rf vim_binding
fi
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding

#jupyter nbextension enable vim_binding/vim_binding

# Generate default Jupyter configuration file.
jupyter notebook --generate-config -y
