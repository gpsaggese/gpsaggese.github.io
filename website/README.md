# GP Saggese Personal Website

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
  > ./test.sh      # Run local server
  > ./publish.sh   # Deploy to GitHub Pages
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

- Follow the formatting rules in `/Users/saggese/src/umd_classes1/helpers_root/docs/ai_prompts/blog.format_rules.md`

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
├── publish.sh                    # Deployment script
└── test.sh                       # Local testing script
```

## Configuration

- Site and blog configuration is in `mkdocs.yml`
- Blog plugin settings include pagination, categories, read time, and more
- SEO meta tags for social sharing are in `overrides/main.html`
- Author information is in `docs/.authors.yml`
