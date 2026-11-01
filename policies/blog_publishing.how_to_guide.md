# How to Review and Publish a Blog or Tutorial

- This is the end-to-end flow to take a draft blog from
  `website/docs/blog/posts/draft.*` to a published post on the website,
  Substack, and LinkedIn
- Without a process, drafts stay at 90% forever; this doc exists so they ship
- Related docs:
  - [`website/README.blog.md`](../website/README.blog.md): list of published
    blogs, blogs ready for review, and drafts ranked by readiness
  - [`website/README.md`](../website/README.md): how the website is built, front
    matter fields, categories, formatting rules
  - [`pr_process.how_to_guide.md`](pr_process.how_to_guide.md): how to create a
    PR (branches, commits, testing section)
  - [`My_AI_Policy.md`](../website/docs/blog/posts/My_AI_Policy.md) and
    [`FAQ.md`](FAQ.md): AI policy and authorship policy
  - Issue [#697](https://github.com/gpsaggese/gpsaggese.github.io/issues/697):
    how to contribute to blogs
  - [`all.write_blog.how_to_guide.md`](../helpers_root/docs/blogging/all.write_blog.how_to_guide.md)
    and
    [`all.publish_blogs.how_to_guide.md`](../helpers_root/docs/blogging/all.publish_blogs.how_to_guide.md):
    shared blog writing conventions, and the publishing setup of the Causify blog
    (different from this website)
  - [Blog Scripts](#blog-scripts): what each blog file in `website/` does

## Overview

| Step | Who              | Output                                       |
| :--- | :--------------- | :------------------------------------------- |
| 1    | Author           | Blog picked from `README.blog.md`            |
| 2    | Author           | GitHub issue from the template below         |
| 3    | Author           | Row in the publishing gsheet                 |
| 4    | Author           | PR with the improvements                     |
| 5    | GP + 2 reviewers | PR reviewed and merged (still a `draft.`)    |
| 6    | Author + GP      | Publication week set in the gsheet           |
| 7    | Author           | Substack entry drafted                       |
| 8    | Author + GP      | Published on website, Substack, LinkedIn     |

## Blog Scripts

- All the files are in `website/`; run them from the repo root
- `README.blog.md`: the blog tracker, with published blogs, blogs ready for
  review, and drafts ranked by readiness
  - Read it in step 1 to pick a blog; it is refreshed in step 8
- `format_blog.sh`: runs `prettier` on the given files, wrapping prose at 80
  columns, in place
  - Needs `prettier` (`npm install -g prettier`)
  - Used in step 4
- `mark_blog_as_ready.py`: publishes a draft with `git mv` (removes the `draft.`
  prefix), sets `draft: false`, and sets `date:` to today
  - `--undo` turns a published blog back into a draft
  - The file must already be committed to git
  - It overwrites `date:` with today, so run it on the publication day
  - Used in step 8
- `find_published_blogs.sh`: prints the blogs with `draft: false`, newest first,
  as markdown bullets
  - Used to refresh `## Published Blogs` in `README.blog.md`
- `prompt.update_README_blog.md`: instructions for Claude Code to refresh
  `README.blog.md`
  - Checks the website builds, checks that `draft:` matches the file name,
    reruns `find_published_blogs.sh`, and rebuilds the drafts table
  - Used in step 8

## 1. Pick a Blog

- Open the `# Drafts` table in `website/README.blog.md`
  - Start from the top: blogs at 80-90% are the fastest to ship
  - Pick a topic you know or can test end-to-end
- Check the gsheet and the
  [Blogs_Tutorials project](https://github.com/users/gpsaggese/projects/12) that
  nobody else is already working on it
- What "improve it" means (from #697):
  - Propose more ideas and fill the gaps
  - Edit and improve the text
  - Make it visually appealing, e.g., add diagrams and plots
  - Run every command and notebook and fix what is broken
  - Add more examples to explain the ideas
- As always, no AI slop: you must be able to explain every line

## 2. Create an Issue

- Title: `Blog: <Title of the blog>`
- Add it to the `Blogs_Tutorials` project
- Use this template as the body:

  ```markdown
  ## Blog
  - File: `website/docs/blog/posts/draft.<name>.md`
  - Current readiness (from `README.blog.md`): XX%

  ## What needs to be done
  - [ ] ...
  - [ ] ...

  ## Checklist
  - [ ] Issue added to the gsheet
  - [ ] ETA: YYYY-MM-DD
  - [ ] Added myself as an author (`.authors.yml` and front matter)
  - [ ] Ran every command / notebook in the blog and they work
  - [ ] Scratch notes at the end turned into content or removed
  - [ ] No TODOs left in the file
  - [ ] Formatted and linted
  - [ ] `mkdocs build --strict` passes
  - [ ] Previewed on the local website
  - [ ] PR: #...
  - [ ] Reviewer 1: @...
  - [ ] Reviewer 2: @...
  - [ ] PR merged
  - [ ] Publication week set in the gsheet
  - [ ] Substack entry created
  - [ ] Published on website / Substack / LinkedIn
  ```

- Post an update in the issue every 2-3 days: done, next, blockers

## 3. Add It to the Google Sheet

- Add a row to the
  [publishing gsheet](https://docs.google.com/spreadsheets/d/1EXprKVLcm6rkqaxM4GYhthGNq3YQwgMzr1ORf7c8C1I/edit)
  with: title, file path, issue link, author(s), status, ETA
- Keep the status in sync with the issue as you move through the steps

## 4. Do a PR With the Improvements

- Follow [`pr_process.how_to_guide.md`](pr_process.how_to_guide.md) for
  branch, worktree, and commits
  - Branch name: `UmdTask<issue>_<Short_title>`
- Set the file you work on:
  ```bash
  > FILE=website/docs/blog/posts/draft.<name>.md
  ```
- Keep the `draft.` prefix in the file name and `draft: true` in the front
  matter
  - The blog is published in step 8 with `mark_blog_as_ready.py`
  - Draft files are excluded from the site by `exclude_docs` in
    `website/mkdocs.yml`
- Check the front matter (the YAML block between `---` lines at the top of the
  blog) against "Front Matter Fields" in
  [`website/README.md`](../website/README.md), plus `draft: true`
- Add yourself as an author
  - Add an entry in `website/docs/.authors.yml`
  - Add your id to `authors:` in the blog front matter
  - Do not leave `TBD` in `url:` or `avatar:`: it renders as a broken link
  - Adding someone else as an author needs GP's approval
- Before asking for review, run the content checks:
  - Run every command and notebook and paste the result if it changed
  - Clean up the scratch notes: drafts often end with raw notes, links, or
    command dumps from GP
    - Turn the useful ones into content, e.g., a "Further reading" or "Other
      tools to explore" section
    - Remove the rest; ask GP in the issue if unsure whether a note is still
      needed
  - No TODOs left
    ```bash
    > cc -p "/coding.todoai_gp $FILE"
    ```
  - Remove AI slop
    ```bash
    > cc -p "/blog.humanize $FILE"
    ```
  - Add links to related blogs
    ```bash
    > cc -p "/blog.add_links $FILE"
    ```
  - Format the text
    ```bash
    > website/format_blog.sh $FILE
    ```
  - Render the images, if any
    ```bash
    > render_images.py -i $FILE
    ```
  - Check the site builds with no warnings
    ```bash
    > (cd website && mkdocs build --strict)
    ```
  - Preview how it renders
    ```bash
    > open_md.py --input $FILE --mode github
    > website/preview_website.sh
    ```
- Open the PR
  - Title: `Blog: <Title of the blog>`
  - Add it to the `Blogs_Tutorials` project, with GP and the 2 reviewers as
    reviewers
  - Use the sections from
    [`pr_process.how_to_guide.md`](pr_process.how_to_guide.md), with this blog
    checklist:
    ```markdown
    Closes #<issue>

    ## Summary
    - Blog: `website/docs/blog/posts/draft.<name>.md`
    - What changed: <text, examples, plots, fixed commands>
    - Authors added: <ids> (in `.authors.yml` and front matter)

    ## Testing
    - [ ] Ran every command / notebook: <on what input>
    - [ ] `website/format_blog.sh` and `pre-commit run --files <files>`
    - [ ] `mkdocs build --strict` passes
    - [ ] Previewed with `website/preview_website.sh`, including author profiles

    ## Pending
    - <What is still open, e.g., GP's review, publication date, a failing check>

    ## Design choices
    - <Non-obvious choices, and why>

    I used Claude Code to help write and test this, and I reviewed every change.
    ```
  - If a GitHub check fails, find out whether your change caused it and say so
    in the PR
  - E.g., [#819](https://github.com/gpsaggese/gpsaggese.github.io/pull/819) for
    a publish PR

## 5. Review and Merge

- GP and 2 other people from the team review the blog in the PR
- What the reviewers check:
  - Is it correct? Do the commands and code actually work?
  - Is it clear to someone who does not know the topic?
  - Is the TL;DR accurate and is the title specific?
  - Is the tone sober and technical (no click-bait, no AI slop)?
  - Are the diagrams and plots readable?
- Address every comment with new commits on the same branch, and resolve the
  threads
- The PR is merged when it is a clear improvement, even if the blog is not
  ready to publish yet
- Update the issue checklist and the gsheet

## 6. Schedule It

- We publish one blog per week
- Pick a free week in the gsheet, so two blogs do not go out the same week
- The `date:` in the front matter is set on the publication day by
  `mark_blog_as_ready.py` (step 8)

## 7. Create the Substack Entry

- Create a draft on Substack with:
  - The title and TL;DR from the blog
  - The body, adapted to Substack (check code blocks, images, and links)
  - A link back to the post on the website
- Leave it as a draft and link it in the issue

## 8. Publish

- On the publication day, open a publish PR
  ```bash
  > website/mark_blog_as_ready.py --file $FILE
  > (cd website && mkdocs build --strict)
  ```
  - Title: `Blog: Publish <Title of the blog>`
  - The script renames the file with `git mv`, so commit the rename
- After the PR is merged, deploy the website
  ```bash
  > website/publish_website.sh
  ```
- Publish the Substack entry and post on LinkedIn with a link to the blog
- Refresh `website/README.blog.md`
  ```bash
  > ccp "Execute website/prompt.update_README_blog.md"
  ```
- Check for dead links
  ```bash
  > uvx linkchecker https://gpsaggese.github.io/ 2>&1 | tee linkchecker.log
  ```
- Close the issue and mark the row as published in the gsheet
