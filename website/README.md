# GP's Personal Website

This is a personal website built with MkDocs Material, featuring a blog
integrated into the main site.

## Installation

- Install MkDocs and the Material theme with blog support:
  ```bash
  > pip install -r requirements.txt
  ```

  Or manually:
  ```bash
  > pip install mkdocs>=1.5.3 mkdocs-material>=9.5.0 pymdown-extensions>=10.7
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

- Or use the convenience scripts:
  ```bash
  > ./test.sh              # Run local server against docs/ as-is (fastest)
  > ./preview_website.sh   # Rebuild Jupyter Books, then run local server
  > ./publish_website.sh   # Rebuild Jupyter Books, then deploy to GitHub Pages
  ```

### Website Generation Flow

The site has two kinds of content:

1. **Hand-written pages**: `docs/*.md`, `docs/blog/posts/*.md`, etc. MkDocs
   renders these directly, so editing them and running `mkdocs serve` (or
   `./test.sh`) is enough to see the change.

2. **Generated content**, copied as static files under `docs/` and then
   served/deployed by MkDocs like any other file:
   - **Jupyter Books** (`docs/jupyter_books/<book>/`): built from the
     `data605`, `msml610`, and `tutorials` course dirs by
     `./publish_jupyter_books.sh` (MyST `jupyter-book build --html`, output
     copied in). Rebuilds are incremental: each book is skipped if none of
     its source notebooks/markdown changed since its last publish (pass
     `--force` to rebuild anyway, or `--books <name,...>` to limit which
     books are considered). Not in `mkdocs.yml`'s `nav:`, so these pages are
     reachable only by direct link.
   - **Class links** (`docs/class_links/<course>.links.html`): built from
     `data605`, `msml610`, `book_springer` by `./update_class_links.sh`
     (wraps `class_scripts/publish_class_links.py`). Also not in `nav:`.

`preview_website.sh` and `publish_website.sh` both call
`publish_jupyter_books.sh` first so the Jupyter Books are current, then run
`mkdocs serve --clean --open` / `mkdocs gh-deploy --no-history` respectively.
`test.sh` skips that step and serves `docs/` as-is — use it when you only
changed hand-written pages and don't need the (rarely-changing) generated
content refreshed. `update_class_links.sh` is run manually, on demand, when
lesson files change.

End-to-end, from a source edit to the live site:

```
edit docs/*.md ─────────────────────────────┐
                                              ├─▶ mkdocs serve/gh-deploy
edit data605|msml610|tutorials notebooks ──▶ publish_jupyter_books.sh ──▶ docs/jupyter_books/*
edit lesson files ──▶ update_class_links.sh ──▶ docs/class_links/*.html ──┘
```

## Blog

The website includes an integrated blog under the "Blog" tab. Blog posts are
located in `docs/blog/posts/`.

### Writing Blog Posts

#### Creating a New Post

1. Create a new Markdown file in `docs/blog/posts/` with a descriptive name (e.g., `My_Post_Title.md`)

2. Add the required front matter at the top:
   ```markdown
   ---
   title: "Your Blog Post Title"
   authors:
     - gpsaggese
   date: YYYY-MM-DD
   description: Brief description for SEO
   categories:
     - Category Name
   ---

   TL;DR: Your punchy one-liner summary.

   <!-- more -->

   Your blog content starts here...
   ```

#### Front Matter Fields

- **title**: Use double quotes, capitalize major words
- **authors**: List format with username(s) from `docs/.authors.yml`
- **date**: Use YYYY-MM-DD format
- **description**: Brief description for SEO and social media
- **categories**: Choose from allowed categories

#### Available Categories

- AI Research
- Machine Learning
- Deep Learning
- Software Engineering
- Startup
- Teaching
- Data Science
- Python

#### Formatting Guidelines

- Follow the formatting rules in `/Users/saggese/src/umd_classes1/helpers_root/docs/ai_prompts/blog.format.md`

- Key formatting rules:
  - Always include `<!-- more -->` tag after TL;DR to separate excerpt
  - Use `##` for main sections, `###` for subsections
  - Use `-` for bullet lists
  - Bold important terms with `**text**`
  - Use inline code for technical terms with `` `code` ``
  - Include blank lines between sections

#### Features

The blog supports:
- **Categories and tags** - Organize posts by topic
- **Reading time estimates** - Automatically calculated
- **Pagination** - 10 posts per page
- **Archive by year** - Automatically generated
- **Author profiles** - Configured in `docs/.authors.yml`
- **Math equations** - Via MathJax
- **Code syntax highlighting** - Multiple languages supported
- **Mermaid diagrams** - For flowcharts and diagrams
- **Social sharing** - Open Graph and Twitter cards via SEO overrides

## Structure

```
website/
├── docs/
│   ├── blog/
│   │   ├── index.md              # Blog landing page
│   │   └── posts/                # Blog posts go here
│   ├── class_links/               # Generated: see update_class_links.sh
│   ├── jupyter_books/             # Generated: see publish_jupyter_books.sh
│   ├── assets/                   # Images, logos, favicon
│   ├── javascripts/              # Custom JavaScript (MathJax)
│   ├── stylesheets/              # Custom CSS
│   ├── .authors.yml              # Blog author configuration
│   ├── index.md                  # Home page
│   ├── 02_cv.md                  # CV/Resume tab
│   ├── 03_education.md           # Education tab
│   ├── 04_teaching.md            # Teaching tab
│   ├── 05_publications.md        # Publications tab
│   ├── 06_research.md            # Research tab
│   └── 07_coding.md              # Coding tab
├── overrides/
│   └── main.html                 # SEO meta tags for social sharing
├── mkdocs.yml                    # Site and blog configuration
├── requirements.txt              # Python dependencies
├── test.sh                       # Serve docs/ as-is, no regeneration
├── preview_website.sh            # Rebuild Jupyter Books, then serve locally
├── publish_website.sh            # Rebuild Jupyter Books, then deploy to GH Pages
├── publish_jupyter_books.sh      # Build/copy Jupyter Books into docs/jupyter_books/
├── update_class_links.sh         # Regenerate docs/class_links/*.html
├── format_blog.sh                # Run prettier over blog post(s)
└── find_published_blogs.sh       # List non-draft blog posts by date
```

## Configuration

- Site and blog configuration is in `mkdocs.yml`
- Blog plugin settings include pagination, categories, read time, and more
- SEO meta tags for social sharing are in `overrides/main.html`
- Author information is in `docs/.authors.yml`
