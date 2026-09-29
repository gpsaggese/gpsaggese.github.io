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

- **Milestone 1: build the static analysis layer**
  - Use `astroid` (or `jedi`) to build a name/type-resolved function-level
    call graph that follows `self.foo()` and imported-name calls to their
    real definitions, not just raw `ast` edges
  - Use `grimp` to build the module-level import graph for the
    package/module abstraction layer
  - This is the result: a Python library that, given a target codebase,
    returns a structured graph with nodes for modules/classes/functions
    and edges for imports/calls

- **Milestone 2: render multi-level static diagrams**
  - Map the graph into a nested file -> class -> function structure, and
    render module-level and file-level views with Graphviz `cluster_*`
    subgraphs
  - Add class-level diagrams (inheritance, composition) reusing
    `pyreverse`-style UML output
  - This is the result: static Graphviz diagrams at three abstraction
    levels (module, file, class) for a sample codebase

- **Milestone 3: build the interactive click-to-follow explorer**
  - Build a `dash-cytoscape` app using compound nodes for the file ->
    class -> function nesting, with expand/collapse and click-to-drill-down
  - Use `libcst` to keep exact source spans so each node can show its
    real source snippet, not just a structural label
  - This is the result: a running interactive web app that lets a user
    click through a real codebase from module down to function, seeing
    source code at each level

- **Milestone 4: animate interactions and validate on a real codebase**
  - Add animation of a call trace or execution path over the graph (e.g.,
    highlighting nodes/edges in sequence as a chosen entry point runs)
  - Run the full pipeline on `helpers_root` or another module of this
    repo as the target codebase
  - This is the result: a demo animating a call trace over the graph of a
    real repo module, with a short write-up of what the visualization
    reveals about that module's structure

## References
- Author(s), _Title_. (Year)
