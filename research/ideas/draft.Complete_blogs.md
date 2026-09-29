# Complete Blogs

## Status
**Status:**: draft
**Complete Specs:**: 0-100%
**Assignee:**: ...

## Core Idea

- There are lots of blog ideas related to AI/ML in:
  - `website/README.blog.md`
  - `website/docs/blog/posts/`
- Pick one or two that are interesting and finish them, including doing
  evaluation, etc.

## Formalization

- [Mathematical notation, definitions, or pseudocode]

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- **Milestone 1: inventory and select candidate blogs**
  - Read `website/README.blog.md` and every draft in
    `website/docs/blog/posts/` and list all incomplete posts
  - Score each candidate on how much technical substance already exists,
    whether it has code/results that can be evaluated, and its relevance
    to current DATA605/MSML610 material
  - This is the result: a ranked shortlist of 1-2 candidate blogs to
    complete, with the selection rationale written down

- **Milestone 2: complete the technical content and evaluation**
  - Fill in missing sections of the chosen blog(s), run and verify every
    code example, and produce (or rerun) the evaluation the post promises
    (benchmarks, plots, comparison tables)
  - This is the result: a complete draft with working code and generated
    evaluation artifacts (tables/plots) for each chosen blog

- **Milestone 3: polish with the existing blog skills pipeline**
  - Run `blog.humanize`, `blog.add_links`, and `blog.add_visuals` on each
    draft, then run `blog.check_format` and fix any violations
  - This is the result: a publish-ready blog post passing
    `blog.check_format` for each of the 1-2 chosen ideas

- **Milestone 4: final review**
  - Proofread each post's claims against the generated evaluation results,
    and verify every code snippet and link works end to end
  - This is the result: 1-2 finished, reviewed blog posts ready to merge
    into `website/docs/blog/posts/`

## References
- Author(s), _Title_. (Year)
