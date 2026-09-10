### [x] Document the agentic auto_task engineering flow

* Repo: umd_classes

* Problem
- We built an agentic workflow (`auto_task` rules, template, skills) to run AI
  coding agents end to end (spec -> branch -> PR) with minimal supervision, but
  it is not documented anywhere outside the skill files themselves
- We want 3 blog posts that explain the flow: what we have today and what we
  plan to have once it is done

* Solution

- [x] PR1: Write `blog_posts/draft.My_agentic_engineering_flow.md`
  - Find and review all the relevant info in the repo before writing
  - Describe the available pieces of info in the repo:
    - `.claude/skills/auto_task.rules.md` (conventions for creating, queuing,
      executing an `auto_task`)
    - `.claude/templates/auto_task.template.md` (problem/solution/PR plan format)
  - Describe the tools and the workflow:
    - Create a `tasks.md` (e.g., from `msml610/book/prompt.slides_and_book_flow.md`)
      in the auto_task format
    - Review it with `/auto_task.criticize tasks.md`
    - Execute it with `/auto_task.execute_with_stacked_prs tasks.md` or
      `/auto_task.execute_interactively tasks.md`
  - Describe the auto_task skills (`mdm skill l auto_task`):
    `auto_task.create_specs_from_todos`, `auto_task.criticize`,
    `auto_task.execute_interactively`, `auto_task.execute_with_stacked_prs`
  - Explain the `/pr.*` skills (`pr.get_ci_to_pass`, `pr.get_local_tests_to_pass`,
    `pr.get_to_commit_state`)
  - Cover both current state (what is implemented and used today) and planned
    state (what is still missing / on the roadmap)

- [x] PR2: Finish `blog_posts/draft.how_to.A_queue_of_AI_coding_agents.md`
  - This is the canonical version; `blog_posts/draft.A_queue_of_AI_coding_agents.md`
    is an older duplicate and is not touched by this task
  - Complete the draft describing the async queue-of-agents workflow

- [x] PR3: Finish `blog_posts/draft.how_to.Stacked_PRs_for_agentic_developent.md`
  - Complete the draft describing the stacked-PR workflow for agentic development
