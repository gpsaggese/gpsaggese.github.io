# Goal
- Update `website/README.blog.md`

# Workflow

## Step 1

- Make sure the blog can be rendered by running `website/preview_website.sh` and fix
  the problems following `.claude/skills/blog.rules.md`

## Step 2

- Make sure the blogs in `website/docs/blog/posts/` have the `draft` property in sync
  with their title
  - E.g., a blog post whose file starts with `draft.` should have `draft: true`,
    a blog post not starting with `draft.` should have `draft: false`
  - Report the violations to the user and ask them to fix by using the `draft: VALUE`
    in the frontmatter as ground truth
  - Use `/git.move` to rename the file and update the references

## Step 3

- Run the script `website/find_published_blogs.sh` to update the list of
   `## Published Blogs` in `website/README.blog.md`

## Step 4

- Update a table of all the draft blogs in `website/docs/blog/posts/draft*`
   - Write the result in a markdown table with columns
     - File: with full path
       - E.g., `website/docs/blog/posts/draft.in_10_mins.helpers_open_md.md`
     - Number of Words
     - Ready %: in 0 to 100%
     - Comment: less than 50 chars on what needs to be done to get it to 100%
   - Rank them from how close they are to be publishable

