# How We Work on Tukey

- Process: `helpers_root/docs/planning/all.plan_work.how_to_guide.md`

## Resources

- Issue [#628](https://github.com/gpsaggese/gpsaggese.github.io/issues/628);
  research Google Doc: to be linked from #628
- Plan: [`04-plan.md`](04-plan.md); log: [`07-progress.md`](07-progress.md),
  one history row per day of work

## Rules

- Work only on tasks in the plan; one PR per plan entry, after the previous review
- Each PR has an owner who opens it and can explain every line; each owner
  reviews one other component
- Every code PR reruns the full tests and the Docker build before it opens
- Using AI is fine, but the author must be able to explain every line; the
  progress log states how AI was used
- GitHub for technical discussion, Slack for quick pings, office hours for the rest
- Decisions go in [`03-decisions.md`](03-decisions.md), never only in Slack;
  record only checks that were actually run
