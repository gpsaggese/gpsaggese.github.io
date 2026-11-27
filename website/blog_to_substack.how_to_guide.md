# How to Convert a MkDocs Blog to Substack

- This is how to copy a blog post from the website
  (`website/docs/blog/posts/*.md`) to our Substack
- Substack is a mirror of the website: it promotes our blogs and collects
  subscribers
  - The website stays the source of truth: fix a post there, then copy it again
  - Each Substack post keeps the original date and credits the original authors
- Related docs:
  - [`blog_publishing.how_to_guide.md`](blog_publishing.how_to_guide.md): the
    end-to-end flow for a blog; this doc details its step 7 (Substack)
  - Issue [#698](https://github.com/gpsaggese/gpsaggese.github.io/issues/698):
    how this process was worked out
  - Issue [#870](https://github.com/gpsaggese/gpsaggese.github.io/issues/870):
    converting all the existing blogs

# Overview

| Step | What                                       | Output                          |
| :--- | :----------------------------------------- | :------------------------------ |
| 1    | Check the post is live on the website      | Post and images online          |
| 2    | Convert the post with `blog_to_substack.py` | HTML on the clipboard or a page |
| 3    | Paste into a new Substack article          | Substack draft                  |
| 4    | Check the draft                            | Draft ready to publish          |
| 5    | Publish                                    | Post on Substack                |
| 6    | Track it                                   | Row updated in the gsheet       |

# Why Not Copy-Paste by Hand

- The Substack editor keeps formatting only from HTML on the clipboard; pasted
  Markdown is not converted
- The other routes we tried break in ways that are slow to fix by hand:

| Route                                  | Result                                                  |
| :------------------------------------- | :------------------------------------------------------ |
| Open the HTML in a browser, copy, paste | Bullets, bold, and italics are lost                     |
| HTML → Google Docs → Substack           | Text is fine, but code blocks lose the code box and line breaks, and YAML lines starting with `-` turn into bullets |
| Clean HTML put on the clipboard         | Headings, lists, bold, inline code, code blocks, links, and images are kept |

- The script does the last route, with the fixes below
- E.g., [How to Use OpenRouter with Simonw's LLM CLI](https://ameersohail.substack.com/p/how-to-use-openrouter-with-simonws)
  was converted this way

# Setup

- Run everything from the repo root
- Install the tools once
  ```bash
  > brew install pandoc
  > pip install pyyaml pymdown-extensions
  ```
  - `pandoc` 3 or newer is needed for the tables
  - `pyyaml` and `pymdown-extensions` are already there if you can build the
    website
- For tables as images (`--tables png`), also install Playwright
  ```bash
  > pip install playwright
  > playwright install chromium
  ```
- Ask GP to add you to the Substack team
  - **Contributor**: can write and publish posts
  - **Admin**: can also change Settings and see Stats and Subscribers

# 1. Check the Post Is Live on the Website

- Substack downloads images from their link when you paste, so the post and its
  images must be on https://gpsaggese.github.io/blog/ first
  - GP deploys the website with `website/publish_website.sh`
- The script checks this and prints a `WARNING` for anything that is not online
- Open the live post and skim it: if it renders badly there, fix the Markdown
  first

# 2. Convert the Post

- Copy the body straight to the clipboard (macOS, or Linux with `xclip`)
  ```bash
  > website/blog_to_substack.py --file website/docs/blog/posts/how_to.Use_OpenRouter.md --copy
  ```
  - It prints the title, subtitle, and date to fill in Substack
- Or build a page with Copy buttons, e.g., to work through many posts
  ```bash
  > website/blog_to_substack.py --file website/docs/blog/posts/how_to.Use_OpenRouter.md
  > website/blog_to_substack.py --all
  ```
  - Open `tmp.build/substack/<name>.html` (or `all_posts.html`) in Chrome
  - Each post has **Copy title**, **Copy subtitle**, **Copy body**, and
    **Preview**
- What the script changes in the post and why:

| Change                                                           | Why                                                                 |
| :--------------------------------------------------------------- | :------------------------------------------------------------------ |
| Adds "Originally published on gpsaggese.github.io by ..." at the top | Credits the authors by name and GitHub handle, and links the original post |
| Drops the front matter, HTML comments, and `<div>` tags (keeps their text) | Substack ignores raw HTML                                           |
| Code blocks become plain `<pre><code>`                           | Substack shows them as code boxes                                   |
| Links whose text is a full `https://` URL are shown without `https://` | Such links make the paste fail when the HTML goes through Google Docs |
| A list is split at each code block                               | Substack moves code blocks out of lists, leaving an empty bullet before the sub-bullets |
| Image paths point to the live website                            | Substack can only fetch images from a public link                    |
| Tables become aligned text in a code block (default)             | Substack has no tables                                              |
| `$` is kept as text, not read as math                            | E.g., `$0.03-$0.05` lost its `$` signs                               |

## Tables

- `--tables code` (default): each table becomes aligned text in a code box
  - Always works; wide tables scroll sideways on phones
- `--tables png`: each table becomes an image, styled like a simple table
  ```bash
  > website/blog_to_substack.py --file website/docs/blog/posts/how_to.Count_cost.md --tables png
  ```
  - The images are embedded in the copied HTML and saved in
    `tmp.build/substack/<name>.tables/table_<N>.png`
  - The table text is kept as the image's alt text
  - If Substack drops an embedded image, upload the PNG with the image button
    in the same spot

# 3. Paste Into Substack

- Open the publication's dashboard and click **Create → Article** (not
  **Note**, which is a short social post)
- Fill in:
  - **Title**: the post's `title`
  - **Subtitle**: the post's `description`, if any
  - **Body**: click "Start writing..." and paste with Cmd+V
    - Wait a few seconds for long posts
    - If nothing is pasted, reload the page (Cmd+R) and paste again
- Byline: see [Authors](#authors)

# 4. Check the Draft

- Click **Preview** (not **Continue**) and scroll the whole post
  - [ ] The "Originally published" line is at the top, with working links
  - [ ] Headings, bullets, and nested bullets look right
  - [ ] Code blocks are in grey boxes with the right line breaks
  - [ ] Images are shown; replace any "Image not found" with an upload
  - [ ] Tables are readable
  - [ ] Nothing is missing at the end
- Set the date: post settings (bottom right in the editor) → date and time at the
  bottom → the post's `date`
  - Set the year first, or the change may not stick

# 5. Publish

- Click **Continue** to open the publish screen
  - **Delivery**: "Send via email and the Substack app" is checked by default
    - Old posts copied from the website: uncheck it, so subscribers do not get
      a burst of emails for posts that are not new
    - New posts on their publication day: leave it checked
  - **Scheduling**: leave it unchecked to publish now
- Click **Publish**
- Open the post in a private window to check that anyone can see it

# 6. Track It

- In the [publishing gsheet](https://docs.google.com/spreadsheets/d/1EXprKVLcm6rkqaxM4GYhthGNq3YQwgMzr1ORf7c8C1I),
  add the Substack link to the post's row and mark it published
- For a new blog, also tick "Substack entry created" in its issue (see
  [`blog_publishing.how_to_guide.md`](blog_publishing.how_to_guide.md))

# Authors

- Substack shows the people who wrote a post as its byline, while our authors are
  the GitHub handles in the post's `authors:` and in `website/docs/.authors.yml`
- The script always adds the "Originally published ... by" line with the authors
- For the byline itself, publish as the publication's own account and remove
  your own name from the byline (the ✕ on the author chip under the title)
  - Pending: GP to confirm which Substack account to use for the byline

# Known Issues

- Images that are not on the live website must be uploaded by hand, e.g., the
  images of `how_to.Render_md_from_terminal.md` live in a `draft.*.figs` folder,
  which the website build skips (`exclude_docs` in `website/mkdocs.yml`)
- Raw HTML in a post (other than comments and `<div>` tags) is not converted
