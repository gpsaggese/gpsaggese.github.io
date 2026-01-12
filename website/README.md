# GP Saggese Personal Website

## Installation

- Install MkDocs and the Material theme:
  ```bash
  > pip install mkdocs mkdocs-material
  ```

## Usage

### Local Development Server

- To run the website locally with live-reload:
  ```bash
  > cd website
  > mkdocs serve
  ```

- The website will be available at `http://127.0.0.1:8000`

### Build Static Site

- To build the static HTML files:
  ```bash
  > cd website
  > mkdocs build
  ```

The built site will be in the `site/` directory.

### Deploy to GitHub Pages

- To deploy directly to GitHub Pages:
  ```bash
  > cd website
  > mkdocs gh-deploy
  ```
