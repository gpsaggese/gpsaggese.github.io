# Code Visualization for Python Codebases

## Status
**Status:**: draft
**Complete Specs:**: 0-100%
**Assignee:**: ...

## Core Idea

- Given a Python code base, build a tool to visualize the code:
  - Parse the code to see how functions and objects interact
  - Animate the interactions, and let the user click to follow the code
  - Support different levels of abstraction
- Build a representation, in Python, of a Python code base:
  - Show how functions relate to each other in a visual form, through a
    graph
  - Correlate the structure of the code with the code of each function
  - Use multiple levels of abstraction: file level, module level

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

- Parsing and static analysis:
  - **`libcst`**: like `ast`, but preserves the exact formatting/whitespace
    and gives a full concrete syntax tree; better for showing/highlighting
    actual source snippets per node rather than just structural facts
  - **`astroid`** (what `pylint` is built on): does type/name inference, so
    it can resolve `self.foo()` or imported names to their actual
    definitions much better than raw `ast`; this is usually the difference
    between a call graph that is roughly right and one that is actually
    usable
  - **`jedi`**: static analysis engine with go-to-definition / find
    references built in; handy for resolving cross-file/cross-module call
    edges without writing a custom resolver
  - **`grimp`**: purpose-built for building import graphs at package/module
    level (used by `import-linter`); a good fit for the module-level layer
    of the abstraction stack
- Call / dependency graph generation:
  - **`code2flow`**: generates function-level call graphs across a whole
    codebase, outputs to Graphviz DOT or JSON; a good starting point/
    reference implementation
  - **`pydeps`**: module-level dependency graphs via Graphviz
  - **`pyreverse`** (bundled with `pylint`): UML-style class diagrams
    (inheritance, composition)
- Visualization:
  - **Graphviz** (`graphviz` or `pygraphviz` Python bindings): `cluster_*`
    subgraph blocks map naturally to file/module grouping; best for static,
    clean hierarchical diagrams
  - **`dash-cytoscape`** (Cytoscape.js in Dash): supports compound nodes
    (nodes containing nodes), which matches the file -> class -> function
    nesting, plus expand/collapse and click-to-drill-down; likely the best
    fit for genuinely multi-level interactive exploration
  - **`pyvis`**: quick interactive HTML graphs (vis.js), less structured
    than Cytoscape but very fast to get something clickable

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Author(s), _Title_. (Year)
