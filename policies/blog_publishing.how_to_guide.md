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

## Overview

| Step | Who               | Output                                         |
| :--- | :-------------    | :--------------------------------------------- |
| 1    | Author            | Blog picked from `README.blog.md`              |
| 2    | Author            | GitHub issue from the template below           |
| 3    | Author            | Row in the publishing gsheet                   |
| 4    | Author            | PR with the improvements                       |
| 5    | GP + 2 reviewers  | PR reviewed and merged (still a `draft.`)      |
| 7    | Author + GP       | Publication date set in the gsheet             |
| 8    | Author            | Substack entry drafted                         |
| 9    | GP                | Published on website, Substack, LinkedIn       |

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
  - Adding more examples to help convey ideas better
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
  - [ ] No TODOs left in the file
  - [ ] Formatted and linted
  - [ ] Previewed on the local website
  - [ ] PR: #...
  - [ ] PR merged
  - [ ] Reviewer 1: @...
  - [ ] Reviewer 2: @...
  - [ ] Publication date set in the gsheet
  - [ ] Substack entry created
  - [ ] Published on website / Substack / LinkedIn
  ```

- Post an update in the issue every 2-3 days: done, next, blockers

## 3. Add It to the Google sheet

- Add a row to the
  [publishing gsheet](https://docs.google.com/spreadsheets/d/1EXprKVLcm6rkqaxM4GYhthGNq3YQwgMzr1ORf7c8C1I/edit)
  with: title, file path, issue link, author(s), status, ETA
- Keep the status in sync with the issue as you move through the steps

## 4. Do a PR With the Improvements

- Follow [`pr_process.how_to_guide.md`](pr_process.how_to_guide.md) for
  branch, worktree, and commits
  - Branch name: `UmdTask<issue>_<Short_title>`
- Keep the `draft.` prefix in the file name and `draft: true` in the front
  matter
  - Do not rename the file to publish it: that happens in step 9 with
    `mark_blog_as_ready.py`
  - Draft files are excluded from the site by `exclude_docs` in
    `website/mkdocs.yml`
- Add yourself as an author
  - Add an entry in `website/docs/.authors.yml`
  - Add your id to `authors:` in the blog front matter
- Check the front matter (see `website/README.md` for the allowed categories)

  ```yaml
  ---
  title: "Your Blog Post Title"
  draft: true
  authors:
    - gpsaggese
    - <your_id>
  date: YYYY-MM-DD
  description: One sentence for SEO and social cards
  categories:
    - <Category>
  ---

  TL;DR: One-line summary.

<!-- more -->
  ```

- Before asking for review, run the content checks:
  - Run every command and notebook and paste the result if it changed
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
  - Preview how it renders
    ```bash
    > open_md.py --input $FILE --mode github
    > website/preview_website.sh
    ```
- In the PR description, under `Testing`, list what you ran and on what input,
  e.g., "ran all 4 examples on a sample PDF, formatted with `format_blog.sh`,
  previewed the website"

## 5. Review and Merge

- GP reviews the PR; address every comment and resolve the threads
- The PR is merged as soon as it is a clear improvement, even if the blog is
  not ready to publish yet
- Update the issue checklist and the gsheet

## 6. Content Review by 2 People

- Ask 2 other people from the team to read the blog
  - Tag them in the issue, and ask them to comment there or on a follow-up PR
- What the reviewers check:
  - Is it correct? Do the commands and code actually work?
  - Is it clear to someone who does not know the topic?
  - Is the TL;DR accurate and is the title specific?
  - Is the tone sober and technical (no click-bait, no AI slop)?
  - Are the diagrams and plots readable?
- Fix the comments in a follow-up PR

## 7. Schedule It

- Pick a publication date in the gsheet
  - We publish about one blog per week, so check the calendar for a free slot
- Set the same `date:` in the front matter

## 8. Create the Substack Entry

- Create a draft on Substack with:
  - The title and TL;DR from the blog
  - The body, adapted to Substack (check code blocks, images, and links)
  - A link back to the post on the website
- Leave it as a draft and link it in the issue

## 9. Publish

- Mark the blog as ready: this renames the file (removing `draft.`) and sets
  `draft: false`
  ```bash
  > website/mark_blog_as_ready.py --file $FILE
  ```
- Preview, fix problems, and publish the website
  ```bash
  > website/preview_website.sh
  > website/publish_website.sh
  ```
- Publish the Substack entry and post on LinkedIn with a link to the blog
- Update `website/README.blog.md`
  ```bash
  > ccp "Execute website/prompt.update_README_blog.md"
  ```
- Check for dead links
  ```bash
  > uvx linkchecker https://gpsaggese.github.io/ 2>&1 | tee linkchecker.log
  ```
- Close the issue and mark the row as published in the gsheet


