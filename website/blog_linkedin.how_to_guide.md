# How to Post a Blog on LinkedIn

- Use this to promote a published blog on LinkedIn
- Each LinkedIn post is a short version of the blog, with links to the website
  and to the Substack post
  - The website stays the source of truth: the LinkedIn post only points to it
- The goal is to measure engagement (readers, impressions, clicks) and grow it
  toward targets, e.g., 1,000 and then 2,000 readers (see GP in #817)
- Related docs:
  - [`blog_publishing.how_to_guide.md`](blog_publishing.how_to_guide.md): the
    end-to-end flow for a blog; this doc details the LinkedIn part of its step 8
  - [`blog_to_substack.how_to_guide.md`](blog_to_substack.how_to_guide.md): the
    Substack post that the LinkedIn post links to
  - Issue [#817](https://github.com/gpsaggese/gpsaggese.github.io/issues/817):
    the weekly audience report
  - Issue [#877](https://github.com/gpsaggese/gpsaggese.github.io/issues/877):
    milestones (LinkedIn outreach is Milestone 3)
  - Issue [#936](https://github.com/gpsaggese/gpsaggese.github.io/issues/936):
    how this process is being worked out

# Overview

| Step | What                   | Who            | When                          |
| :--- | :--------------------- | :------------- | :---------------------------- |
| 1    | Check the blog is live | LinkedIn owner | Before drafting               |
| 2    | Write the post         | LinkedIn owner | Before the review             |
| 3    | Review the post        | GP             | Before posting                |
| 4    | Post it                | Org account    | Same day as the Substack post |
| 5    | Share it               | Team           | Same day                      |
| 6    | Track the numbers      | LinkedIn owner | Weekly, for the report        |

# 1. Check the Blog Is Live

- The LinkedIn post links to the blog, so make sure the blog is online first
  - The blog on https://gpsaggese.github.io/blog/ (GP deploys it weekly)
  - The Substack post (see
    [`blog_to_substack.how_to_guide.md`](blog_to_substack.how_to_guide.md))
- Open both links and check that they load

# 2. Write the Post

- Use this template
  ```text
  <Hook: 1-2 lines with the problem or the most surprising result>

  <1 sentence on what the blog covers>

  - <Key point 1>
  - <Key point 2>
  - <Key point 3>

  Read it on the website: <website URL>?utm_source=linkedin
  Or on Substack: <Substack URL>

  #<Hashtag1> #<Hashtag2> #<Hashtag3>
  ```
- **Hook**: LinkedIn shows only the first lines before "see more", so put the
  most interesting point there
- **Key points**: 3-5 points; use only facts and numbers that are in the blog
- **Links**: always link both the website and Substack
  - `?utm_source=linkedin` on the website link lets Google Analytics count the
    clicks that come from LinkedIn
- **Image**: the blog's main figure or diagram, if any
- **Hashtags**: based on the blog's `categories:`
- **Tone**: the same as the blog, no clickbait
- **Length**: LinkedIn allows up to 3,000 characters; aim for much less

# 3. Review the Post

- Post the draft as a comment in the blog's issue before posting
- Tag GP for review, and apply his comments

# 4. Post It

- Post on the same day as the weekly Substack post
- The account that posts is to be confirmed by GP in
  [#936](https://github.com/gpsaggese/gpsaggese.github.io/issues/936)
- Attach the image, and check that both links work in the published post
- Link the LinkedIn post in the blog's issue

# 5. Share It

- The team reshares the post on the same day
- Add one line of your own when resharing, instead of a bare repost

# 6. Track the Numbers

- Every week, add the numbers to the LinkedIn tab of the
  [weekly report](https://docs.google.com/spreadsheets/d/19n2gV8IZ3JnFicclOnglCl9qvzNA6o4S7hGWpeTNV50/edit?usp=sharing)
  - Impressions, reactions, comments, and reposts, from the post's analytics
  - Clicks from LinkedIn to the website, from Google Analytics
    (`utm_source=linkedin`)
- How we get the LinkedIn analytics is to be confirmed by GP in
  [#936](https://github.com/gpsaggese/gpsaggese.github.io/issues/936)

# Checklist

- Copy this into the blog's issue
  ```markdown
  ## LinkedIn

  - [ ] Blog live on the website and on Substack
  - [ ] Post drafted from the template, as a comment in this issue
  - [ ] Reviewed by GP
  - [ ] Posted on the publication day: <LinkedIn link>
  - [ ] Reshared by the team
  - [ ] Numbers added to the LinkedIn tab of the weekly report
  ```
