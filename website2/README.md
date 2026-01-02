# GP Saggese Personal Website

This is a MkDocs-based website for GP Saggese's personal and professional profile.

## Prerequisites

- Python 3.7+
- pip

## Installation

Install MkDocs and the Material theme:

```bash
pip install mkdocs mkdocs-material
```

## Usage

### Local Development Server

To run the website locally with live-reload:

```bash
cd website2
mkdocs serve
```

The website will be available at `http://127.0.0.1:8000`

### Build Static Site

To build the static HTML files:

```bash
cd website2
mkdocs build
```

The built site will be in the `site/` directory.

### Deploy to GitHub Pages

To deploy directly to GitHub Pages:

```bash
cd website2
mkdocs gh-deploy
```

## Project Structure

```
website2/
├── mkdocs.yml          # MkDocs configuration file
├── docs/               # Source documentation files
│   ├── index.md        # Home page (profile)
│   ├── cv.md           # CV/Resume page
│   ├── publications.md # Publications page
│   ├── gp_publications.bib  # BibTeX file
│   └── images/         # Image assets
│       ├── gp-saggese-headshot.png
│       ├── gp-saggese-2012.png
│       └── gp-saggese-2022.png
└── site/               # Generated static site (after build)
```

## Customization

Edit `mkdocs.yml` to customize:
- Site metadata (name, description)
- Theme settings (colors, features)
- Navigation structure
- Plugins and extensions

## Theme

This website uses the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme with:
- Dark/light mode toggle
- Search functionality
- Navigation tabs and sections
- Social links (LinkedIn, GitHub, Email)

## Content

- **Home**: Professional profile and biography
- **CV/Resume**: Detailed career history and achievements
- **Publications**: Academic publications with citations and links
