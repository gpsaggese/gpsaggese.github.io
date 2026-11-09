// git_hash=0b9e24c3-53e timestamp=20261106_182025
// Import AIMA style formatting and macros.
#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (
  aima-style, algorithm, chapter, glossary, styled-table,
)
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": (
  cite, references,
)

// Document metadata
#set document(
  title: "Lesson 02.01: Skills Organization and Management",
  author: "AI Software Engineering",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("Lesson 02.01: Skills Organization and Management")

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:7 '* Roadmap'
// Slide: Roadmap
#strong[Roadmap]

Code models complete patterns. A repository that wants consistent code from an
agent has to state its conventions in files that both engineers and agents can
find, and this chapter is about those files.

A #emph[topic] groups three kinds of agent files: rules, templates, and skills.
Each kind answers one question, and each fact has exactly one home. The
#emph[agent instruction manifest] (`CLAUDE.md`, `AGENTS.md`) is the entry point:
every session loads it, and it points the agent to the rules.

The second half of the chapter treats skills as code. Writing a skill turns a
workflow that the team repeats into a `SKILL.md` with frontmatter, inputs,
workflow steps, and a verification section. Maintaining skills means learning
from Git history, checking that every reference still resolves, and keeping the
skill set small. Testing skills borrows from software testing: sample runs,
golden outputs, assertions that tolerate stochastic output, and regression
checks after a rule changes.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:22 '# Why Repositories Need Agent Conventions'
// Slide: Why Repositories Need Agent Conventions
#strong[Why Repositories Need Agent Conventions]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:24 '* Why Agents Need Explicit Conventions'
// Slide: Why Agents Need Explicit Conventions
#strong[Why Agents Need Explicit Conventions]

An AI model completes patterns: given no stated convention, it invents plausible
but inconsistent code. It might write `dict` where the rest of the repository
uses `Dict[str, str]` everywhere, reach for `pathlib.Path` where the team has
standardized on `os.path`, or produce a Google-style docstring in a project that
follows REST style. Each choice is individually reasonable, yet the result
clashes with what already exists.

A new human engineer faces the same problem on day one. The difference is that
the engineer reads the style guide, copies the structure of a nearby file, and
asks a teammate when something is ambiguous. An AI agent, by contrast, starts
each session with no memory of the last one and has no teammate to ask. Every
conversation is, in effect, a fresh first day.

The solution is to make the codebase itself the style guide. An
#strong[AI-optimized codebase] states its conventions in files that both humans
and agents can find and follow: naming patterns, import preferences, docstring
format, directory layout. Most properties that shorten onboarding for a human
(clear naming, consistent structure, explicit contribution guides) apply equally
to an agent. When the rules live in the repo rather than in tribal knowledge,
the agent can read them at the start of every session and produce code that
fits.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:44 '* Restating Conventions per Session Does Not Scale'
// Slide: Restating Conventions per Session Does Not Scale
#strong[Restating Conventions per Session Does Not Scale]

A naive first answer is to restate every convention in the prompt at the start
of each session. That approach fails for several reasons: it burns context
window on every run, drifts out of sync with the actual codebase (since nothing
ties the prompt text to the repository), cannot be reviewed or audited (because
it lives only in a transient chat log), and differs from one engineer to the
next, since each person pastes a slightly different version.

The fix is to check the conventions into the repository as versioned
files. A #strong[manifest] (such as `CLAUDE.md`) serves as the entry point that
every session loads automatically. Alongside it sit #strong[rule files] encoding
the conventions for each topic (naming, formatting, architecture) and
#strong[skills] capturing the step-by-step procedures the team repeats. As
@fig:restatingconventionspersessiondoesnotscale shows, pasted prompts drift
apart across sessions, while convention files committed to the repo reach every
session unchanged.

// rendered_images:begin
// ```graphviz
// digraph PerSessionVsRepo {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.25;
//     ranksep=0.35;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.6,
//           fontname="Helvetica", fontsize=12, margin="0.16,0.08",
//           height=0.40];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // prompt : orange | session : blue | repo files : violet note
//     // drift : red
// 
//     subgraph cluster_naive {
//         label="Pasted per session"; labelloc="t";
//         fontname="Helvetica-Bold"; fontsize=12; fontcolor="#3A4A5C";
//         style="rounded,filled"; fillcolor="#F7F9FB"; color="#C7D0DA";
//         penwidth=1.0; margin=12;
// 
//         PA [label="Prompt v1",
//             fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//         PB [label="Prompt v2",
//             fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//         SA [label="Session A",
//             fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//         SB [label="Session B",
//             fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//         { rank=same; PA; PB; }
//         { rank=same; SA; SB; }
//     }
// 
//     PA -> SA;
//     PB -> SB;
//     PA -> PB [style=bold, penwidth=2.0, color="#B23A48",
//               fontcolor="#B23A48", label="  Drift", constraint=false];
// 
//     subgraph cluster_repo {
//         label="Checked into the repo"; labelloc="t";
//         fontname="Helvetica-Bold"; fontsize=12; fontcolor="#3A4A5C";
//         style="rounded,filled"; fillcolor="#F7F9FB"; color="#C7D0DA";
//         penwidth=1.0; margin=12;
// 
//         PR [label="Pull request",
//             fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//         Files [label="CLAUDE.md, rules, skills", shape=note, style=filled,
//                fillcolor="#E9E1F7", color="#A88FD9", fontcolor="#45296B"];
//         Every [label="Every session",
//                fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     }
// 
//     SA -> PR [style=invis];
//     PR -> Files [label="  Reviewed"];
//     Files -> Every [label="  Loaded"];
// }
// ```
// label=fig:restatingconventionspersessiondoesnotscale
// caption=Pasted prompts drift apart, while files in the repo reach every session
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.1.png",
    width: 70%,
  ),
  caption: [Pasted prompts drift apart, while files in the repo reach every session],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:restatingconventionspersessiondoesnotscale>
// render_images:end
Conventions stored in the repository are reviewed, diffed, and versioned exactly
like production code. When agent behavior needs to change, that change flows
through a pull request: it is proposed, discussed, approved, and merged with the
same rigor as any other code change. This closes the loop between how the team
works and how the agent works, making the two evolve together rather than apart.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:133 '* Consistency as a Reliability Property'
// Slide: Consistency as a Reliability Property
#strong[Consistency as a Reliability Property]

#strong[Consistency] means that the same kind of problem is solved the same way
everywhere in the codebase. #strong[Reproducibility] means that the same code,
configuration, and data produce the same result on any machine, at any time.
Reliable software engineering rests on both.

Conventions built around consistency and reproducibility speed up development
considerably. A reader who knows the convention can guess the answer to a
question about the codebase instead of searching for it. When every module
follows the same naming scheme, the same directory layout, and the same
error-handling pattern, onboarding a new contributor or debugging an unfamiliar
service becomes a matter of pattern recognition rather than archaeology.

Site reliability engineering treats consistency as a reliability property, not a
style preference #cite("beyer2016site"). A #emph[runbook] turns a procedure that
lives in one engineer's head into exact steps that anyone on call can execute
under pressure. A #emph[checklist] for a deploy catches the step that a tired
engineer would skip #cite("gawande2009checklist"). Both work because they enforce a single, repeatable way of doing things: the on-call
engineer at 3 a.m. does not improvise, and the deploy pipeline does not silently
skip a migration.

The same principle applies to coding agents. Rules, skills, and manifests are
the runbooks and checklists of an AI-assisted development workflow. They encode
project-wide conventions in a form the agent can follow on every run, so the
code it writes stays consistent with the rest of the codebase, whichever
engineer starts the session.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:156 '# Topics, Rules, Templates, and Skills'
// Slide: Topics, Rules, Templates, and Skills
#strong[Topics, Rules, Templates, and Skills]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:158 '* Topics Group Three Kinds of Files'
// Slide: Topics Group Three Kinds of Files
#strong[Topics Group Three Kinds of Files]

A #strong[topic] is a domain that groups related conventions and skills. Each
topic is named with a singular, lowercase word: for instance, `coding`,
`testing`, `notebook`, `slides`, or `skill`. Topics serve as the primary
organizational unit for everything an AI workflow needs to know about a
particular area of work.

Each topic owns up to three kinds of files. The first is a #emph[rule file],
stored at `.claude/skills/<TOPIC>.rules.md`, which captures the standing
conventions for that domain. The second is a #emph[template], stored at
`.claude/templates/<TOPIC>.template.<EXT>`, providing a starter file in the
appropriate format. The third is a set of #emph[skills], individual procedure
files stored under `.claude/skills/<TOPIC>.<ACTION>/SKILL.md`, each describing
one concrete task the agent can perform within the topic.

For scale, this repository contains 165 skills, 33 rule files,
and 38 templates. The `coding` topic alone includes `coding.rules.md`,
`coding.template.py`, and 21 skills under various `coding.*` action directories.
@fig:topicsgroupthreekindsoffiles shows how a single topic like `coding` groups
its skills, rule file, and template.

// rendered_images:begin
// ```graphviz
// digraph FourKinds {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.25;
//     ranksep=0.45;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.8,
//           fontname="Helvetica", fontsize=12, margin="0.18,0.10",
//           height=0.45];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // skill : teal | rule file : blue | template : orange
// 
//     subgraph cluster_coding {
//         label="Topic: coding"; labelloc="t";
//         fontname="Helvetica-Bold"; fontsize=12.5; fontcolor="#3A4A5C";
//         style="rounded,filled"; fillcolor="#F7F9FB"; color="#C7D0DA";
//         penwidth=1.0; margin=16;
// 
//         S1 [label="coding.fix_docstring",
//             fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//         S2 [label="coding.fix_comments",
//             fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//         S3 [label="coding.rename",
//             fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//         Rule [label="coding.rules.md",
//               fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//         Tpl [label="coding.template.py",
//              fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//         { rank=same; S1; S2; S3; }
//     }
// 
//     S1 -> Rule [label="  Points to"];
//     S2 -> Rule;
//     S3 -> Rule;
//     Rule -> Tpl [style=dashed, color="#8C8C8C", label="  Shown in"];
// 
//     subgraph cluster_legend {
//         label=""; style="rounded,filled"; fillcolor="#F7F9FB";
//         color="#C7D0DA"; penwidth=1.0; margin=14;
//         node [fontsize=10, height=0.3, margin="0.10,0.05"];
//         L1 [label="How", fillcolor="#B7DDD0", color="#6FA890",
//             fontcolor="#1F4E39"];
//         L2 [label="Why and when", fillcolor="#9CC4F2", color="#3C6FB0",
//             fontcolor="#1F4E79"];
//         L3 [label="What to copy", fillcolor="#FFC98A", color="#D98E2B",
//             fontcolor="#6B4517"];
//         { rank=same; L1; L2; L3; }
//     }
//     Tpl -> L2 [style=invis];
// }
// ```
// label=fig:topicsgroupthreekindsoffiles
// caption=A topic groups skills (teal), a rule file (blue), and a template (orange)
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.2.png",
    width: 70%,
  ),
  caption: [A topic groups skills (teal), a rule file (blue), and a template (orange)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:topicsgroupthreekindsoffiles>
// render_images:end
// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:242 '* Rules: Why and When'
// Slide: Rules: Why and When
#strong[Rules: Why and When]

A #strong[rule file] (`<TOPIC>.rules.md`) holds the conventions for a single
topic: principles, decision criteria, and short good/bad examples that make each
convention concrete. Related rules sit together under one top-level section,
with a separate `##` header for every individual rule. Those headers are stable
identifiers: both `rig --rule` (which retrieves a rule by name) and
`cc_lint.py --rule` (which applies just that rule section to a file) rely on
them to locate the right block. Rule files tend to be long; `coding.rules.md`,
for instance, runs to roughly 1,470 lines.

A rule states the convention and then shows it:

```markdown
## Use REST Style for Docstrings
- Always use REST style for docstrings
  - **Bad**: ...
  - **Good**: ...
```

This pattern keeps each convention self-contained: the header names it, the body
explains it, and the examples remove any ambiguity about what compliance means.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:266 '* Templates: What to Copy'
// Slide: Templates: What to Copy
#strong[Templates: What to Copy]

A #strong[template] is a starting-point file, named `<TOPIC>.template.<EXT>`,
that fixes the boilerplate every new file of that kind needs: headers, imports,
section markers, and `<VAR>` placeholders ready to be filled in. Where a rule
describes what good output looks like in words, a template shows what good
output looks like as an actual file you can open and read. The two are
complementary: rules carry the reasoning ("why this choice beats that one"),
while templates carry the concrete structure ("here is the skeleton, fill in the
blanks").

Several templates recur throughout this workflow:

- `coding.template.py` for production `.py` files
- `testing.template.py` for `test_*.py` test modules
- `slides.template.md`, `project_prd.template.md`, and
  `github_PR_plan.template.md` for documentation and planning artifacts

A single template serves two audiences at once. For a human author, it is
boilerplate to copy and customize, saving the minutes otherwise spent recreating
the same header block and section outline from memory. For an LLM, it is a
reference style: the model can use it both as a target format when generating or
transforming a file, and as a checklist when reviewing an existing file for
structural violations. A template never explains why one structural choice beats
another; that kind of decision criterion belongs in the corresponding rule file,
not in the skeleton itself.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:291 '* Skills: How, Step by Step'
// Slide: Skills: How, Step by Step
#strong[Skills: How, Step by Step]

A #strong[skill] is a named, versioned procedure for a single task, stored at a
path like `.claude/skills/<TOPIC>.<ACTION>/SKILL.md`. Examples include
`testing.write_unit_tests`, `docker.shrink_container`, and
`coding.fix_docstring`. Each skill captures one repeatable workflow in a form
that an AI coding agent can execute on demand.

Skills activate either through an explicit request (for instance,
`/coding.fix_docstring hio.py`) or automatically when a task matches the skill's
description. A skill directory can bundle scripts, reference files, and other
assets alongside its instruction file. A skill always points to the
rule file for its topic rather than restating the rules inline; this keeps
knowledge in one place and avoids drift between copies.

`coding.fix_docstring` shows how thin a skill can be. Its entire instruction amounts to: "Read and apply to `<FILES>` the
rules from `coding.rules.md` in `## Use REST Style for Docstrings`." The
convention itself, what REST-style docstrings look like, lives in the rule file.
The skill says only #emph[what to do] with that convention: apply it to the
specified files.

The underlying design principle is separation of concerns: a skill adds the
#emph[procedure] (when to run, on which files, in what order), while the rule
file keeps the #emph[knowledge] (what the standard actually requires). Because
many skills can point to the same rule file, updating a convention in one place
propagates to every workflow that depends on it.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:315 '* Rules, Templates, and Skills Compared'
// Slide: Rules, Templates, and Skills Compared
#strong[Rules, Templates, and Skills Compared]

When you need to add a new piece of reusable knowledge to a project's `.claude/`
directory, the first question is what kind of file it should live in.
@tab:rulestemplatesandskillscompared summarizes the three options and the single
question each one answers.

#figure(
  styled-table(
    headers: ("Artifact", "Answers", "Location under .claude/", "Example"),
    rows: (
      (
        "Rule file",
        "Why and when",
        "skills/<TOPIC>.rules.md",
        "coding.rules.md",
      ),
      (
        "Template",
        "What to copy",
        "templates/<TOPIC>.template.<EXT>",
        "coding.template.py",
      ),
      (
        "Skill",
        "How, step by step",
        "skills/<TOPIC>.<ACTION>/SKILL.md",
        "coding.fix_docstring",
      ),
    ),
  ),
  caption: [The three kinds of agent files, the question each answers, and where each lives.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:rulestemplatesandskillscompared>

A #strong[rule file] captures a convention that cuts across many tasks: naming
schemes, formatting standards, import policies, or any constraint an agent
should obey regardless of what it is building. A #strong[skill file] captures a
step-by-step procedure for one specific task, such as "how to convert a slide
deck into a Typst chapter" or "how to run the test suite and interpret its
output." A #strong[template] is a skeleton or worked example meant to be copied
verbatim and filled in: a boilerplate `.typ` file, a starter configuration, or a
reference implementation of a common pattern.

Each kind answers exactly one question, which tells a reader (whether human or
AI agent) which file to open. If you are asking "what style or constraint
applies here?", you want a rule. If you are asking "what are the steps to do
X?", you want a skill. If you are asking "what should the result look like?",
you want a template. Keeping the three kinds separate means no single file tries
to do double duty, and the agent can load only the file relevant to its current
need rather than parsing a monolithic guide for the one paragraph that matters.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:340 '* One Home per Fact'
// Slide: One Home per Fact
#strong[One Home per Fact]

Every documentation base faces a common problem: the same fact ends up recorded
in several places, such as a branch naming pattern appearing in a skill file, a
how-to guide, and a blog post. When that happens, copies drift apart over time.
One gets updated while the others go stale, and two copies that contradict each
other leave an agent (or a human reader) choosing between them at random, with
no way to know which is current.

The solution is to give each fact exactly one canonical home and have every
other place link to it rather than restating it. This is the "don't repeat
yourself" principle applied to documentation #cite("hunt1999pragmatic").

To find that home, walk through the following decision list and stop at the
first match:

1. A convention shared by several skills within a topic belongs in the topic's
  rules file (`<TOPIC>.rules.md`).
2. A procedure that an agent executes for a single task belongs in a skill or
  action file (`<TOPIC>.<ACTION>/SKILL.md`).
3. Content whose value lies in the ordering of steps or in a human judgment call
  belongs in a how-to guide under `docs/`.
4. A story, an opinion, or a lesson learned belongs in a blog post.

Links between these layers flow in one direction only: from the narrative level
toward the precise level. A blog post may reference a how-to guide, and a how-to
guide may reference a skill or rule file, but a skill never depends on a blog
post. The reason is that blogs age fastest: they capture a moment's reasoning or
experience, while skills and rules are kept current as the system evolves.
Pointing a durable skill at a perishable blog would introduce a fragile
dependency that breaks silently the moment the blog falls out of date.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:364 '# Agent Instruction Manifests'
// Slide: Agent Instruction Manifests
#strong[Agent Instruction Manifests]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:366 '* Agent Instruction Manifest: Definition'
// Slide: Agent Instruction Manifest: Definition
#strong[Agent Instruction Manifest: Definition]

An #strong[agent instruction manifest] is a versioned file committed to the
repository that describes architecture, conventions, and canonical commands,
written for a program rather than for a human newcomer. It functions as a
repository-level system prompt: the harness loads it at session start, before
the first user request, so the agent begins every session already aware of
project structure, naming conventions, and preferred workflows.

The same idea appears under different names depending on the tool. `AGENTS.md`
is an open, tool-agnostic format that many coding agents read
#cite("agentsmd2025"). Claude Code uses
`CLAUDE.md`, Cursor stores per-project rules under `.cursor/rules/`, and GitHub
Copilot reads `.github/copilot-instructions.md`. Despite the naming differences,
the purpose is identical: give the agent enough context to act correctly without
asking.

This is not documentation in the traditional sense.
Traditional docs target a human reader who will skim, search, and interpret
ambiguity charitably. An instruction manifest is an operating manual whose first
reader is a machine: it must be precise, unambiguous, and structured so that a
language model can parse it reliably on every cold start.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:389 '* What Goes in a Manifest'
// Slide: What Goes in a Manifest
#strong[What Goes in a Manifest]

An #emph[agent instruction manifest] (e.g., `AGENTS.md` or `CLAUDE.md`) is a machine-readable file that tells a coding agent
everything it needs to know about a repository before it writes its first line
of code. Rather than relying on a back-and-forth onboarding conversation, the
manifest encodes the project's expectations declaratively, so the agent can
orient itself from the moment it opens the codebase.

A well-structured manifest typically contains four kinds of information. First,
an #emph[architecture] section sketches the major components and their ownership
boundaries, giving the agent a map of what lives where. Second, a
#emph[conventions] section spells out module naming, import style, and test
layout so generated code matches the rest of the project. Third, a set of
#emph[canonical commands] lists the blessed way to run tests, lint, and enter
development containers (for example, `invoke run_fast_tests` or
`invoke docker_bash`), eliminating guesswork about the build system. Fourth,
#emph[pointers] direct the agent to the specific rule file that governs each
file type, so it can look up detailed guidance on demand rather than carrying
every rule in the manifest itself.

As a result, the agent becomes productive without an interactive
onboarding loop: it reads the manifest once and already knows the project's
norms. An equally important benefit is transparency for the human team. Because
the manifest lives in the repository alongside ordinary source code, maintainers
change agent behavior through code review rather than editing a hidden prompt
that only one person controls.

The tradeoff is that every line of the manifest consumes context tokens in every
session, whether or not the agent needs that particular fact for the task at
hand. Keeping the manifest lean is therefore a design discipline in its own
right. Two heuristics help. State facts and point to rule files for details
rather than inlining lengthy guidelines: a one-line pointer costs far less
context than a page of formatting rules. Leave out information that changes
often (such as the current sprint's open tasks, which belong in an issue
tracker) and information that a tool already enforces (such as formatting rules
checked by a linter), since restating those in the manifest adds cost without
changing behavior.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:411 '* Layered Manifests: User, Repo, Directory'
// Slide: Layered Manifests: User, Repo, Directory
#strong[Layered Manifests: User, Repo, Directory]

A single manifest cannot serve every scope. A personal preference (such as requesting terse
answers) should apply to every project an engineer touches, while a repo
convention belongs only to that repo, and a library convention should follow the
library into every repo that embeds it.

The harness addresses this by reading and merging manifests from three levels.
First, a #strong[user-level] manifest at `~/.claude/CLAUDE.md` applies to all
projects for one engineer. Second, a #strong[repo-level] `CLAUDE.md` at the
repository root governs everyone working in that repo. Third, a
#strong[directory-level] `CLAUDE.md` in any subdirectory is read whenever the
agent operates inside that directory. The `@path` directive lets any manifest
import an external file, so the manifest itself stays short while still reaching
longer, more detailed documents.

This repository follows the pattern. The user manifest imports a
token-saving style file that keeps responses concise across all projects. The
repo-root `CLAUDE.md` describes the course directory layout and defers to
`helpers_root/CLAUDE.md` for code-related conventions. That helper-level
manifest holds all the build, test, and import rules. The `.claude` directory is
a symlink pointing to `helpers_root/.claude`, which means every repository that
embeds `helpers` as a submodule automatically shares one canonical set of rules
and skill files, with no duplication and no risk of drift between copies.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:437 '* Context Budget: Why Not One Big Manifest'
// Slide: Context Budget: Why Not One Big Manifest
#strong[Context Budget: Why Not One Big Manifest]

A manifest works well while it fits comfortably in the model's context window,
but that comfort has limits. This repository, for instance, carries 33 rule
files totalling roughly 12,000 lines. Loading every one of those files into
every session wastes context tokens on instructions that are irrelevant to the
task at hand and buries the one rule that actually matters under thousands of
lines of unrelated guidance.

A first, naive answer is to collapse everything into one long `CLAUDE.md` that
contains every convention the agent should follow. That fails on two counts: it
costs tokens on every request, even when the user asks for a one-line change
that touches none of those conventions; and instructions buried in the middle of
a long context are followed less reliably than those near the beginning or end
#cite("liu2024lost").

The approach that scales is to load instructions in layers, on demand. A
#strong[manifest] is always loaded and kept short; it is made of pointers rather
than content. A #strong[rule file] is pulled in only when the agent edits a file
of the type that rule governs (a Typst rule file when editing `.typ`, a slides
rule file when editing `.smd`). A #strong[skill] is loaded when a task matches
the skill's description, regardless of file type.

A manifest should function as a whiteboard, not a
library. A whiteboard holds a handful of orienting notes and arrows pointing
elsewhere; a library tries to hold everything. Past a certain size, what an
agent needs is not a longer document but a better index into shorter,
task-relevant ones.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:459 '* Pointers From CLAUDE.md to Rules and Templates'
// Slide: Pointers From CLAUDE.md to Rules and Templates
#strong[Pointers From CLAUDE.md to Rules and Templates]

The manifest needs to stay short, yet it must still guide the agent to every
relevant rule file. The solution is a chain of pointers, each one more specific
than the last. First, `CLAUDE.md` contains a directive: "You MUST read and
strictly follow `.claude/rules.md`." That index file maps each file type to its
dedicated rule file and, where applicable, a template. Each rule file (e.g.
`coding.rules.md`) organizes its constraints under one `##` header per rule.
Finally, a skill definition points to the single rule section it applies. The
agent reads only the rule file for the file type it is
currently editing, keeping its context window lean while still enforcing every
relevant constraint. @fig:pointersfromclaudemdtorulesandtemplates shows
this layered structure: the top-level manifest and index (shown in violet) lead
to the rule file (blue) and the template (orange) of each file type.

// rendered_images:begin
// ```graphviz
// digraph Pointers {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.25;
//     ranksep=0.40;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.8,
//           fontname="Helvetica", fontsize=12, margin="0.18,0.10",
//           height=0.45];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // manifest / index : violet note | rule file : blue | template : orange
//     // skill : teal
// 
//     Claude [label="CLAUDE.md", shape=note, style=filled,
//             fillcolor="#E9E1F7", color="#A88FD9", fontcolor="#45296B"];
//     Index [label=".claude/rules.md", shape=note, style=filled,
//            fillcolor="#E9E1F7", color="#A88FD9", fontcolor="#45296B"];
//     Rule [label="coding.rules.md",
//           fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Tpl [label="coding.template.py",
//          fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Other [label="slides.rules.md, ...",
//            fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
//     Skill [label="coding.fix_docstring",
//            fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
// 
//     Claude -> Index [label="  Always loaded"];
//     Index -> Rule [label="  .py"];
//     Index -> Tpl [label="  .py"];
//     Index -> Other [style=dashed, color="#8C8C8C"];
//     Skill -> Rule [label="  One section"];
//     { rank=same; Rule; Tpl; Other; }
// }
// ```
// label=fig:pointersfromclaudemdtorulesandtemplates
// caption=Manifest and index (violet) lead to rule file (blue) and template (orange)
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.3.png",
    width: 70%,
  ),
  caption: [Manifest and index (violet) lead to rule file (blue) and template (orange)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:pointersfromclaudemdtorulesandtemplates>
// render_images:end
// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:524 '* Enforcing Rules With Hooks'
// Slide: Enforcing Rules With Hooks
#strong[Enforcing Rules With Hooks]

A pointer to a rules file is only a request, not a
guarantee. `CLAUDE.md` is read at session start, but in a long session it drifts
far back in the context window and loses influence. A skill file runs only if
the model happens to select it. Neither mechanism fires reliably at the moment
it matters: right before an edit.

The solution is a #strong[hook]: a command that the harness runs on a fixed
event, not when the model decides to. The file
`.claude/hooks/check_file_rules.sh` runs before every `Write` or `Edit` tool
call. It matches the file path being edited against a table of patterns and
injects a reminder into the conversation, for example: "This is a Python file.
Before proceeding, read and follow `coding.rules.md`." Because the harness
triggers the hook deterministically, the reminder arrives exactly when it is
needed, regardless of how long the session has been running or how far back
`CLAUDE.md` has scrolled. @fig:enforcingruleswithhooks shows this flow:
the hook (shown in teal) fires before every edit (shown in blue), inserting the
rule reminder into the path between the agent's intent and the actual file
write.

// rendered_images:begin
// ```graphviz
// digraph HookFlow {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.30;
//     ranksep=0.35;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.6,
//           fontname="Helvetica", fontsize=12, margin="0.16,0.08",
//           height=0.40];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // agent action : orange | hook : teal | reminder : violet note
//     // edit : blue
// 
//     Call [label="Agent calls Edit(x.py)",
//           fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Hook [label="check_file_rules.sh",
//           fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Note [label="Reminder:\nfollow coding.rules.md", shape=note, style=filled,
//           fillcolor="#E9E1F7", color="#A88FD9", fontcolor="#45296B"];
//     Edit [label="Edit runs",
//           fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
// 
//     Call -> Hook [label="  PreToolUse"];
//     Hook -> Note [label="  Path matches"];
//     Note -> Edit;
//     Hook -> Edit [style=dashed, color="#8C8C8C", label="  No match"];
// }
// ```
// label=fig:enforcingruleswithhooks
// caption=The harness runs the hook (teal) before every edit (blue)
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.4.png",
    width: 70%,
  ),
  caption: [The harness runs the hook (teal) before every edit (blue)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:enforcingruleswithhooks>
// render_images:end
This approach has real limits. It covers only the file types that a path pattern
can identify: `.py`, `test_*.py`, `.sh`, `.ipynb`, `.typ`, and so on. A blog
post and a README are both `.md` files, so purpose-based rules (one set for blog
prose, another for documentation) cannot be distinguished by extension alone and
still require manual prompting. The hook also informs rather than blocks: it
reminds the model to follow the rules but does not prevent the edit from
proceeding if it ignores them. Any check that must not be forgotten belongs on a deterministic trigger, not on the model's voluntary
recall.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:592 '* Machine-Readable Contracts: `repo_config.yaml`'
// Slide: Machine-Readable Contracts: `repo_config.yaml`
#strong[Machine-Readable Contracts: `repo_config.yaml`]

When an agent needs to push a Docker image, it faces two immediate questions:
which image name should it use, and which container registry should it target?
If these facts live only in prose documentation or scattered comments, the agent
is forced to guess, and guessing in a deployment pipeline is a recipe for silent
failures.

A #strong[machine-readable contract] solves this by encoding the operational
facts of a repository in a structured file that agents and scripts can query
directly. Rather than parsing natural-language instructions, the agent reads a
YAML (or JSON) file and extracts exactly the fields it needs. This differs from the agent manifest: a manifest tells
the agent #emph[how to behave] (which tools to use, what workflow to follow),
while a contract tells the agent #emph[facts about the system] (the repository
name, the Docker image name, the container registry URL). The two are
complementary: the manifest governs process, the contract supplies data.

#figure(
  [
    ```yaml
    repo_info:
      repo_name: helpers
    docker_info:
      docker_image_name: helpers
    container_registry_info:
      ghcr: ghcr.io/causify-ai
    ```
  ],
  caption: [A machine-readable repository contract in YAML.],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:repoconfigyaml>

A practical example is a `repo_config.yaml` file like the one shown in
@fig:repoconfigyaml. It declares the repository name, the Docker image name, and
the container registry endpoint in a handful of lines. This file replaced an
earlier executable `repo_config.py`, converting the configuration from code that
had to be imported and run into plain data that any tool, in any language, can
parse without side effects. Data files are also easier to lint,
diff, and validate in CI than executable configuration scripts.

A release task that needs to tag and push an
image reads the registry URL and image name from the contract, constructs the
full image reference, and proceeds. No heuristic matching, no
environment-variable guessing, no fragile string extraction from a README. The
contract is the single source of truth, and every consumer of that truth,
whether human, script, or autonomous agent, reads the same file.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:627 '# Writing a Skill'
// Slide: Writing a Skill
#strong[Writing a Skill]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:629 '* From a Repeated Workflow to a Skill'
// Slide: From a Repeated Workflow to a Skill
#strong[From a Repeated Workflow to a Skill]

When the same prompt keeps appearing in your workflow, it may be time to capture
it as a reusable skill. Several signs point to a prompt being worth that
investment: you have typed it (or something very close to it) for the third
time; you find yourself making the same correction after every agent run; the
task has a clear, verifiable outcome (for instance, tests that pass or a linter
that stays clean); or the task applies uniformly across many files, such as
reformatting every docstring in a package.

Consider a concrete scenario: you repeatedly ask the agent to "fix the
docstrings of this file to our REST style." Each time, either you paste the
style rules into the prompt yourself, or the agent guesses a different
convention and you correct it afterward. The fix is to write a rule section
titled "Use REST Style for Docstrings" that spells out the convention once, then
pair it with a skill called `coding.fix_docstring` that applies those rules on
demand. From then on, nobody pastes the style, and the agent applies the same
convention on every run.

You do not have to start from scratch. The `mdm skill list <topic>` command
searches existing skills for one you can extend rather than duplicate. If none
fits, `mdm skill edit <topic>.<action>` scaffolds a new skill file with the
standard folder layout already in place. For a more guided path, the
`skill-creator` skill itself walks you through questions about your workflow and
writes the skill folder for you.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:652 '* Anatomy of a SKILL.md'
// Slide: Anatomy of a SKILL.md
#strong[Anatomy of a SKILL.md]

A `SKILL.md` file has several parts. The
frontmatter carries a `description` and a `model` field but omits `name`, since
the skill's name is derived from its directory name (`<TOPIC>.<ACTION>`). A
concise goal statement explains what the skill does in one or two sentences. The
inputs section lists every argument as `<VAR>` and specifies fallback behavior
when an argument is missing. The workflow section walks through each step under
its own `##` header. Conventions and constraints point to the relevant rule
files, and examples illustrate both correct and incorrect usage. Finally, a
verification checklist confirms that the output meets all requirements.

The rules for writing a skill are themselves a rule file, `skill.rules.md`, so
skill authors follow the same kind of file that every other topic uses.

For instance, a minimal skill skeleton follows this shape:

```markdown
---
description: <What it does>
model: haiku | sonnet | opus
---
# Goal
- <1-2 sentences>
# Inputs
- <FILE>: <what it is>
# Workflow
## <Step name>
# Conventions
- Follow <TOPIC>.rules.md
# Verification
- [ ] <Check>
```

Each section serves a distinct purpose: the frontmatter provides metadata for
tool selection and routing, the goal anchors the skill's intent, the inputs and
workflow make the skill reproducible, and the verification checklist closes the
loop by ensuring the result actually satisfies the goal.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:694 '* Progressive Disclosure'
// Slide: Progressive Disclosure
#strong[Progressive Disclosure]

#strong[Progressive disclosure] loads a skill in three levels, each one only
when it is needed #cite("anthropic2025skills"). Rather than dumping every rule into the context window at
once, the system exposes just enough information at each stage for the model to
decide whether it needs more. The first level is the frontmatter: the
skill's directory name and a one-line description, loaded for every skill at
session start. The second level is the full body of `SKILL.md`, loaded only
after the model decides the skill matches the task. The third level covers any external
resources the skill references, fetched on demand during execution.

#figure(
  styled-table(
    headers: ("Level", "Content", "Loaded"),
    rows: (
      (
        "1. Frontmatter",
        "Directory name and description",
        "Always, at session start",
      ),
      (
        "2. SKILL.md body",
        "Goal, workflow, verification",
        "When the skill matches the task",
      ),
      (
        "3. Bundled files",
        "scripts/, references/, assets/",
        "When a step needs them",
      ),
    ),
  ),
  caption: [The three levels of progressive disclosure and when each one loads.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:progressivedisclosure>

@tab:progressivedisclosure summarizes the three levels and when each one enters
the context. The frontmatter is the only part every
skill pays for on every request; the full text is loaded selectively.

The frontmatter costs roughly 30 to 50 tokens per skill. With 165 skills in the
repository, that totals approximately 5,000 to 8,000 tokens of always-on
context, a fraction of what loading every skill's complete text would require.
This budget is small enough that it scales comfortably even as the skill library
grows; adding ten more skills costs a few hundred tokens, not tens of thousands.

Contrast this with an MCP server, which front-loads a complete description of
every tool it exposes. An MCP tool description typically runs several lines: the
function signature, parameter types, and a paragraph of usage notes, all of
which enter the context whether or not the model ever calls that tool. A skill's
frontmatter front-loads a single sentence. The model reads that sentence,
decides whether the skill is relevant to the current task, and only then pulls
in the full body.

This design means the `description` field carries outsized importance. It is the
only text the model sees when it chooses whether to activate a skill, so a vague
or misleading description causes the model either to skip a skill it needs or to
load one it does not. Writing a good description is less about completeness and
more about discrimination: it should make clear, in one clause, exactly which
tasks the skill applies to and which it does not.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:726 '* Frontmatter: Description and Model'
// Slide: Frontmatter: Description and Model
#strong[Frontmatter: Description and Model]

The #strong[frontmatter] is the YAML block at the top of `SKILL.md` and the only
part of a skill loaded in every session. It takes a simple form:

```yaml
---
description: Fix Python Docstrings
model: haiku
---
```

A good `description` field starts with an imperative verb (such as "Fix",
"Write", or "Check"), stays under 80 characters, and names the transformation
rather than merely the topic: "Fix Python docstrings" tells you what the skill
does, while "Docstrings" only tells you what it is about.

The `model` field picks the cheapest model that can handle the task reliably. A
mechanical edit that applies a single rule (for instance,
`coding.fix_docstring`) calls for `haiku`. Multi-step work like `skill.auto_add`
needs `sonnet`. Tasks that require judgment across many rules simultaneously,
such as `skill.check_against_rules`, call for `opus`.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:753 '* Inputs and Workflow Steps'
// Slide: Inputs and Workflow Steps
#strong[Inputs and Workflow Steps]

The #strong[Inputs] section of a skill file names every value the skill needs
from the user. Write each input name in angle-bracket notation (e.g., `<FILE>`,
`<PROMPT>`), never shell-style `$var`. Every input declaration should also state
what happens when that input is missing or ambiguous; for instance, "stop and
ask the user. Do not guess." This prevents the skill from silently inventing a
default that might be wrong.

The #strong[Workflow] section lays out the skill's steps as a sequence of `##`
headings, each starting with an action verb. Avoid numbering them "Step 1",
"Step 2", and so on: unnumbered headings can be reordered without triggering a
cascade of renumbering edits. Write each step in imperative language with
measurable criteria. Vague qualifiers such as "properly" or "nicely" carry no
testable meaning and should be replaced with concrete conditions (e.g.,
"compiles without warnings" or "passes the existing test suite").

The #strong[Conventions] section connects the skill to the project's shared rule
files. Point to the rule file and the specific section by name rather than
restating the rule inline. A restated rule creates a second home for the same
fact, and the two copies inevitably drift apart as one gets updated and the
other does not.

As an illustration, the `skill.auto_add` skill uses four workflow headers that
follow these conventions: "Determine the Changes," then "Read the Prompt and the
Conventions," then "Infer the Rules," and finally "Improve the Prompt." Each
header names an action, none carries a step number, and none restates a rule
that already lives in a shared conventions file.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:777 '* Verification and Provenance'
// Slide: Verification and Provenance
#strong[Verification and Provenance]

An agent can claim a task is finished when the work is actually incomplete or
incorrect. The safeguard against this is a #strong[verification section]: a
concrete checklist the agent executes before reporting success. A typical
verification section lists checks such as confirming that every referenced file
exists, that the output follows the required format, and that related files
remain consistent with one another. Written in markdown with unchecked boxes, it
might look like "Confirm all docstrings in these files follow REST style" or
"Confirm the code logic and behavior are unchanged." The checks are not
aspirational; they are meant to be run, and the task is not done until every box
is ticked.

A related practice is #strong[provenance]: a line placed in every file that a
skill creates, naming the skill responsible for generating it. For instance, a
slide deck might begin with
`// Generated by .claude/skills/slides.write/SKILL.md`. This serves two purposes
at once. A human reader who opens the file can trace it back to the exact
instructions that produced it, and an automated pipeline that needs to
regenerate the file can locate the skill to rerun. Provenance turns every
generated artifact into a self-documenting link back to its own creation
process.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:800 '* Worked Example: `coding.fix_docstring`'
// Slide: Worked Example: `coding.fix_docstring`
#strong[Worked Example: `coding.fix_docstring`]

The skill that fixes Python docstrings takes thirteen lines: frontmatter, one
input placeholder, one processing step, and two verification checks. All its
decisions live in one section of `coding.rules.md`, so the cheap `haiku` model
only has to read and apply the rules under the heading "Use REST Style for
Docstrings." This delegation pattern means that adopting a new docstring
convention requires editing only that one rule section; every skill that
references it picks up the change automatically.

The tradeoff is fragility: if the referenced header is ever renamed, the skill
silently points at nothing and the model loses its instructions. The file also
omits a `# Goal` section and a `# Workflow` section, a structural gap that
`skill.check_against_rules` flags on inspection.

The skill file itself, reproduced here without surrounding fences, reads:

```
---
description: Fix Python Docstrings
model: haiku
---
- I will pass you one or more files
  <FILES>
- Read and apply to <FILES> the rules
  from .claude/skills/coding.rules.md
  in ## Use REST Style for Docstrings
# Verification
- [ ] Confirm all docstrings in
  <FILES> follow REST style
- [ ] Confirm the code logic and
  behavior are unchanged
```

This compact format captures the entire task: accept files, apply a referenced
convention, and verify both style compliance and behavioral preservation.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:840 '* Running a Skill'
// Slide: Running a Skill
#strong[Running a Skill]

A #strong[skill] is a reusable prompt template named `<TOPIC>.<ACTION>` that
tells Claude exactly what to do to one or more files. There are three ways to
invoke one. First, you can issue a slash command inside an interactive session,
which processes one skill against one file and lets you review the changes as
they happen. For example, typing `/coding.fix_docstring helpers/hio.py` at the
Claude prompt applies that single skill to a single file. Second, you can use
the `cc_lint.py` shell wrapper, which supports scripted runs across many files
with a fixed model; a typical invocation passes `--files`, `--skill`, and
`--mode one_shot_with_cc` on the command line. Third, the `cc` wrapper opens a
full interactive session where you can mix several skills and manual edits in
the same conversation.

Skills are not the only way to package a prompt. A #emph[slash command file]
(placed in `.claude/commands/<name>.md`) fires only when you explicitly type
`/<name>`, and it cannot bundle auxiliary files the way a skill can. A
#emph[subagent] (placed in `.claude/agents/<name>.md`) is a separate agent with
its own instruction set: for instance, `code.reviewer` reviews the most recent
`git diff` rather than modifying code. Skills sit between these two: more
structured than a bare command file, less autonomous than a full subagent.

One practical habit worth adopting: commit your work before running any skill.
That way each skill application produces exactly one diff you can review in
isolation, making it straightforward to accept, amend, or revert the change.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:866 '# Maintaining Skills'
// Slide: Maintaining Skills
#strong[Maintaining Skills]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:868 '* Why Skills Drift'
// Slide: Why Skills Drift
#strong[Why Skills Drift]

A skill often depends on files it does not own: a rule header in another
document, a script path, or a template location. When any of those moves, the
skill's reference goes stale. The header gets renamed, the script relocates to a
different directory, or a newer model interprets the same wording differently,
and the skill quietly drifts out of sync with the artifact it was supposed to
invoke.

Consider a concrete case from this repository. On 2026-05-12, the rule file
`coding.rules.md` renamed its section header from "Use REST Style for Comments"
to "Use REST Style for Docstrings." The skill `coding.fix_docstring` still
pointed at the old header. No error was raised; the agent simply had to locate
the rule on its own or guess at its contents. The stale reference was not
corrected until 2026-08-03, nearly twelve weeks later. @fig:whyskillsdrift
traces that timeline, highlighting the window during which the skill silently
operated without the rule it was supposed to follow.

// rendered_images:begin
// ```graphviz
// digraph SkillDrift {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.30;
//     ranksep=0.40;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.6,
//           fontname="Helvetica", fontsize=12, margin="0.16,0.08",
//           height=0.40];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // change : orange | stale state : red | fix : blue
// 
//     Rename [label="2026-05-12\nRule header renamed",
//             fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Stale [label="Skill cites the old header\nNo error raised",
//            fillcolor="#FFB3B3", color="#D64545", fontcolor="#6B1F1F"];
//     Fix [label="2026-08-03\nReference fixed",
//          fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
// 
//     Rename -> Stale;
//     Stale -> Fix [style=bold, penwidth=2.0, color="#B23A48",
//                   fontcolor="#B23A48", label="  12 weeks"];
// }
// ```
// label=fig:whyskillsdrift
// caption=A renamed header leaves the skill stale (red) for 12 weeks
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.5.png",
    width: 70%,
  ),
  caption: [A renamed header leaves the skill stale (red) for 12 weeks],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:whyskillsdrift>
// render_images:end
The telling detail is that nothing failed loudly. There was no compile error, no
broken import, no test turning red. The agent received a dead reference, could
not resolve it, and proceeded anyway. Whether it found the renamed rule through
search or simply improvised is unknown (nobody has checked the session logs from
that window), but either outcome is a degradation: the skill's contract with
its dependency was broken, and no signal made that visible.

The lesson generalizes beyond renamed headers. A skill is, in effect, code that
calls other files. It assembles its behavior from references to rules,
templates, and scripts, just as a function assembles its behavior from calls to
other functions. When a function's callee changes its signature, the compiler or
the test suite catches the mismatch. A skill has no compiler, and most skills
have no tests, so the same class of breakage passes in silence. If skills are to be
reliable, they need the same checks that code gets: validated references, change
history tracking, and tests that exercise the dependency chain end to end.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:930 '* Learning From Git History: `skill.auto_add`'
// Slide: Learning From Git History: `skill.auto_add`
#strong[Learning From Git History: `skill.auto_add`]

Every manual correction you make to an agent's output is, in effect, a training
signal: it reveals a gap between what the prompt asked for and what the task
actually required. The `skill.auto_add` workflow turns that signal into a
lasting prompt improvement, so the next run does not repeat the mistake.

The procedure works as follows. Given a file and the prompt that produced it,
`skill.auto_add` diffs the file from its first commit to its last, examining
each changed hunk to identify what was fixed and what general rule that fix
implies. One-off edits (a corrected typo, an updated date) are dropped, and
rules that say the same thing in different words are merged. What remains is a
minimal, targeted edit to the prompt, applied only after review and approval.
The result is a tight feedback loop: the agent runs, a human spots and corrects
an error, and the correction is distilled back into the prompt that governs
future runs, as @fig:learningfromgithistoryskillautoadd shows.

// rendered_images:begin
// ```graphviz
// digraph AutoAddLoop {
//     bgcolor="transparent";
//     pad="0.15";
//     splines=spline;
//     nodesep=0.30;
//     ranksep=0.35;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.6,
//           fontname="Helvetica", fontsize=12, margin="0.16,0.08",
//           height=0.40];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // skill run : teal | manual fix : orange | updated prompt : blue
// 
//     Run [label="Skill run",
//          fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Fix [label="Manual fix, committed",
//          fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
//     Infer [label="skill.auto_add\ninfers the rule",
//            fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Prompt [label="Prompt updated",
//             fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
// 
//     Run -> Fix -> Infer -> Prompt;
//     Prompt -> Run [style=bold, penwidth=2.0, color="#B23A48",
//                    fontcolor="#B23A48", label="  Next run",
//                    constraint=false];
// }
// ```
// label=fig:learningfromgithistoryskillautoadd
// caption=Each manual fix (orange) becomes a prompt rule (blue)
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.6.png",
    width: 70%,
  ),
  caption: [Each manual fix (orange) becomes a prompt rule (blue)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningfromgithistoryskillautoadd>
// render_images:end
The prompt that generates this book's lesson slides learned this way. On
2026-10-13, one commit added `TODO(gp)` comments by hand to a generated deck,
flagging claims that needed checking. The same commit added the general rule to
the prompt: #emph[when you have a doubt about a claim, add a]
`// TODO(gp): Check ...` #emph[comment]. Every later run inherits the behavior.

Generalize the rule behind the diff, not the
literal strings of the one file you happened to fix. A rule like "add a TODO
comment when a claim is uncertain" applies to every future chapter; a rule like
"add a TODO to line 47 of Lesson 3" applies to nothing else. Each fix, properly
generalized, improves every later run across every file the prompt touches.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:996 '* Checking References: `skill.check_refs`'
// Slide: Checking References: `skill.check_refs`
#strong[Checking References: `skill.check_refs`]

// From: msml610/lectures_source/Lesson00.smd:1 '* Skill: Check References in Skill and Rule Files'

`skill.check_refs` is a validation skill whose
goal is to find every reference in a skill or rule file that points to nothing:
a backticked path to a file that does not exist, or a cited header that is
missing from the file it claims to belong to.

The technique relies on two checks, applied systematically over every `SKILL.md`
and every `<TOPIC>.rules.md` in the repository. First, #emph[file existence]:
every path that appears in backticks is tested against the filesystem. Second,
#emph[header existence]: every cited section header is verified to exist inside
the cited file. A typical verification looks like running `grep -q` for the
exact header text against the target file:

```bash
grep -q '^## Use REST Style for Docstrings$' .claude/skills/coding.rules.md
```

If the header is not found, the check fails for that reference.

The output takes the form of a `cfile` in vim quickfix format, with one line per
violation, followed by a numbered list of problems for the user to pick from and
address. For instance, running the check after a commit might report:

```
.claude/skills/coding.fix_docstring/SKILL.md:8: header
  '## Use REST Style for Comments' not found in coding.rules.md
```

This tells you exactly which file, which line, and which header string failed to
match, so you can fix either the reference or the target heading without
searching manually.

Run this check after every change that touches skill or rule files. References
go stale silently: a renamed heading or a moved file leaves behind a citation
that looks fine until someone tries to follow it. Catching these immediately,
while the edit is fresh, is far cheaper than debugging a broken cross-reference
weeks later.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1024 '* Keeping the Skill Set Small'
// Slide: Keeping the Skill Set Small
#strong[Keeping the Skill Set Small]

Skills and rules accumulate over time, and without active maintenance they drift
toward redundancy. Two skills end up doing almost the same thing under different
names; a rule file grows into a flat list with no internal structure; a new rule
repeats one that already exists somewhere else in the collection.

The solution is a set of skills whose sole purpose is maintaining other skills,
each one asking for approval before making changes. `skill.consolidate_skills`
merges redundant skills on the same topic and renames them to eliminate
duplication. `skill.reorganize_rules` regroups the sections of a rule file and
updates every reference that pointed at the old structure. `rules.add`
inserts a new rule under the correct heading, but only after confirming that no
existing rule already covers the same ground. `skill.check_against_rules` reads a
file and writes a plan listing the violations it finds.

These maintenance tools are themselves skills, so the
system improves through the same workflow it supports. There is no separate
"meta" layer for upkeep: the same process that builds a new capability also
cleans up and tightens the existing ones.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1046 '# Testing Skills'
// Slide: Testing Skills
#strong[Testing Skills]

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1048 '* Why Test a Skill'
// Slide: Why Test a Skill
#strong[Why Test a Skill]

A skill is a program that a stochastic interpreter runs. Unlike a deterministic function,
the same input can produce different outputs across two runs. A rule edit, a
model update, or even a renamed header can silently change the behavior, with no
error raised and no test failure flagged.

Conventional software testing works differently. For ordinary
code, `pytest` runs a fixed input through a function and compares the output
exactly: if the return value changes, the test fails. For a skill, the inputs
are fixed but the outputs vary, so any useful check needs a tolerance, some
notion of "close enough" that absorbs the stochastic variation without masking
genuine regressions.

The goal is to build tests that are repeatable enough to run in continuous
integration and catch a skill regression before the team encounters it in
production. What "correct" means for a skill is not obvious. It could
mean an exact string
match, a structurally valid output (e.g., the JSON parses and has the right
keys), semantic equivalence (the output means the same thing even if worded
differently), or an invariant that must hold across all valid outputs (e.g.,
every figure label referenced in the prose also appears in a figure block). Each
of these defines a different testing strategy, and choosing the right one
depends on what the skill is supposed to guarantee.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1068 '* Smoke Test: Run on a Sample File, Review the Diff'
// Slide: Smoke Test: Run on a Sample File, Review the Diff
#strong[Smoke Test: Run on a Sample File, Review the Diff]

The cheapest way to verify a skill is a #strong[smoke test]:

1. Pick a small sample file that exhibits the problem the skill is meant to fix.
2. Commit the current state of the tree so it is clean.
3. Run the skill on that sample.
4. Review `git diff`: every hunk should be a fix the skill promises, and nothing
  else.
5. Run the file's unit tests to confirm nothing broke.

For instance, running `coding.fix_docstring` on a file containing three
Google-style docstrings should produce exactly three docstrings rewritten in
REST style. Any change to a code line, or a docstring left unconverted, is
grounds for rejection.

This approach catches a broken skill in minutes without requiring any special
tooling. The tradeoff is that it is entirely manual: a single run against a
single model, so it will miss failures that surface only with different inputs
or under a different provider.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1089 '* Fixtures and Golden Outputs'
// Slide: Fixtures and Golden Outputs
#strong[Fixtures and Golden Outputs]

A #strong[skill fixture] stores the skill name, an input file, and one or more
assertions that the output must satisfy. For instance, a fixture for a
docstring-conversion skill might point at a Google-style Python file as input
and declare two checks: one verifying that the non-docstring code lines remain
unchanged, and another comparing the output against a reference file, tolerating
at most 10% drift. In YAML this looks like:

```yaml
skill: coding.fix_docstring
input: fixtures/google_style.py
assertions:
  - {type: invariant, check: code_lines_unchanged}
  - {type: golden, expected: fixtures/rest_style.py, max_drift: 0.1}
```

The second assertion type relies on a #strong[golden output]: a stored reference
that later runs are compared against. When the actual output drifts beyond a
configured threshold, the test flags the result for human review rather than
passing silently. The concept mirrors the `check_string()` golden-file pattern
already used for ordinary code tests; extending it to LLM-generated text simply
adds a drift tolerance because exact string equality is too brittle for
natural-language output.

Beyond fixed golden files, #emph[property-based inputs] probe the skill's
boundaries by feeding it edge cases: an empty file, a very long file,
adversarial text designed to confuse the model. This borrows directly from
property-based testing in software engineering #cite("claessen2000quickcheck"),
where generators produce a wide variety of inputs and the test checks that a
stated invariant holds for every one of them.

One practical convention ties the whole scheme together: version the fixtures
alongside the skill definition itself, so that a change to a skill's `.md`
prompt and an update to its test cases appear in a single diff. This keeps the
two artifacts from drifting apart and makes code review straightforward: anyone
reading the prompt change can immediately see whether the tests still match.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1116 '* Assertions for Stochastic Output'
// Slide: Assertions for Stochastic Output
#strong[Assertions for Stochastic Output]

@tab:assertionsforstochasticoutput lists four kinds of assertion, each with an
example of what it checks in the output of `coding.fix_docstring`.

#figure(
  styled-table(
    headers: ("Assertion", "Checks", "Example for coding.fix_docstring"),
    rows: (
      (
        "Exact match",
        "Output equals the golden output",
        "Small file, one docstring",
      ),
      (
        "Structural",
        "Format, schema, or syntax",
        "Output still parses as Python",
      ),
      ("Invariant", "A property that must hold", "Code lines are unchanged"),
      (
        "Semantic",
        "Same meaning, judged by an LLM",
        "Each docstring describes the same parameters",
      ),
    ),
  ),
  caption: [Kinds of assertion for stochastic output, with an example for `coding.fix_docstring`.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:assertionsforstochasticoutput>

Because LLM outputs are stochastic, the testing strategy must account for
non-determinism without sacrificing reliability. A practical approach is to run
rule-based checks first, since they are cheap and exact: a regex that verifies a
required import line, a character-count bound, or a parse that confirms the
output is still valid Python all run in milliseconds and either pass or fail definitively.
Only after those deterministic gates pass does the framework invoke the more
expensive LLM-judged cases. For those, the test harness repeats each case $N$
times and requires a majority of runs to pass, smoothing out the variance
inherent in any single sample.

The harder problem is that an #strong[LLM-as-judge] has its own error rates
#cite("zheng2023judging"). When a language model grades another model's output, it can produce both
false positives (accepting a flawed output as correct) and false negatives
(rejecting a valid one). The way to calibrate trust in such a judge
is to measure its false-positive and false-negative rates on a set of labeled
cases whose ground truth is already known. Once those rates are quantified, the
test framework can set pass thresholds that account for the judge's own
reliability rather than treating its verdicts as ground truth.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1140 '* Detecting Regressions After Rule Changes'
// Slide: Detecting Regressions After Rule Changes
#strong[Detecting Regressions After Rule Changes]

When a single rule section is referenced by many skills, editing it risks
breaking skills that nobody intended to touch. The solution is to treat rule
files the same way a codebase treats shared libraries: every change triggers a
downstream check before it can land.

The regression-detection procedure works as follows:

1. Find every skill that cites the changed rule file (using `rig` to trace the
  dependency graph).
2. Run `skill.check_refs` to verify that all cross-references still resolve.
3. Rerun the fixtures of each affected skill against the updated rules.
4. Compare the new outputs with the golden (known-good) outputs and flag any
  drift.

@fig:detectingregressionsafterrulechanges shows this pipeline: the teal
verification steps must all pass before the blue merge step proceeds.

// rendered_images:begin
// ```graphviz[width=75%]
// digraph Regression {
//     bgcolor="transparent";
//     pad="0.10";
//     splines=spline;
//     nodesep=0.25;
//     ranksep=0.22;
//     rankdir=TB;
// 
//     node [shape=box, style="rounded,filled", penwidth=1.6,
//           fontname="Helvetica", fontsize=12, margin="0.16,0.08",
//           height=0.38];
//     edge [color="#A3B1C0", penwidth=1.3, arrowhead=vee, arrowsize=0.75,
//           fontname="Helvetica", fontsize=10, fontcolor="#7B8794"];
// 
//     // step : teal | outcome : blue
// 
//     Edit [label="Edit a rule file",
//           fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Find [label="Find the skills that cite it",
//           fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Refs [label="skill.check_refs",
//           fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Run [label="Rerun fixtures",
//          fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Cmp [label="Compare with golden outputs",
//          fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     Merge [label="Merge",
//            fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
// 
//     Edit -> Find -> Refs -> Run -> Cmp;
//     Cmp -> Merge [label="  Pass"];
//     Cmp -> Edit [label="  Drift", style=bold, penwidth=2.0,
//                  color="#B23A48", fontcolor="#B23A48", constraint=false];
// }
// ```
// label=fig:detectingregressionsafterrulechanges
// caption=A rule edit reruns the checks (teal) before the merge (blue)
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson02.01_Skills_Organization_and_Management.typ.figs/Lesson02.01_Skills_Organization_and_Management.7.png",
    width: 75%,
  ),
  caption: [A rule edit reruns the checks (teal) before the merge (blue)],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:detectingregressionsafterrulechanges>
// render_images:end
Two tiers of models keep this practical without blowing through compute budgets.
A cheap, fast model runs on every commit inside CI, catching obvious regressions
in seconds. The full production model serves as a gate before a release,
ensuring that the final golden-output comparison reflects the same model quality
end users will see. The principle is the same as for code: a rule change does not merge until the skills that depend on it pass.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1206 '* Measuring Skill Quality'
// Slide: Measuring Skill Quality
#strong[Measuring Skill Quality]

A binary pass/fail verdict on a skill's output hides how close that skill is to
failing: a prompt that scores 92% today may silently drop to 81% after a minor
model update, yet both rounds register as "pass." Measuring quality requires
looking beyond the binary threshold.

#strong[Quality dimensions of a skill] capture distinct failure modes.
#emph[Consistency] measures score variance across repeated runs with the same
input: a high-variance skill is unreliable even when its mean score looks
acceptable. #emph[Robustness] tracks how much the score shifts when the input is
paraphrased without changing its meaning #cite("sclar2024quantifying").
#emph[Context sensitivity] checks whether injecting irrelevant context into the
prompt degrades performance #cite("shi2023large"). #emph[Instruction-following
  fidelity] asks an LLM judge to assess what fraction of the skill's explicit
rules the output actually satisfies #cite("liu2023gpteval").

These dimensions open several practical applications:

- Compare two versions of a skill (before and after an edit) on a held-out task
  set, using the same quality vector rather than a single aggregate number.
- Detect drift when the underlying model changes: a sudden widening of
  consistency or a drop in robustness signals that the skill needs attention,
  even if the mean score holds steady.
- Supply a reward signal to automated prompt optimization
  frameworks #cite("khattab2024dspy").

Optimizing against a judge introduces a real risk, however: #strong[reward
  hacking], where the system learns to produce outputs that satisfy the judge's
scoring rubric without genuinely completing the task #cite(
  "amodei2016concrete",
). A summary might hit every bullet on the judge's checklist yet omit the
source's central argument. Mitigations include rotating judge prompts, using
multiple independent judges, and periodically validating against human ratings.

Existing tooling already runs evaluations of this kind. Promptfoo, DeepEval, and
OpenAI Evals each provide harnesses for scoring LLM outputs on user-defined
criteria, making it straightforward to operationalize these quality dimensions
without building a custom pipeline from scratch.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1231 '* Summary'
// Slide: Summary
#strong[Summary]

A repository can teach an agent its conventions through files, the same way it
teaches a new human engineer. Rather than
relying on verbal instructions or tribal knowledge passed along in meetings, the
repository itself becomes the authoritative source of "how we do things here."
Every convention, every naming pattern, every structural expectation lives in a
file that both humans and agents can read.

A #emph[topic] groups three kinds of files, each answering a distinct
question. A #emph[rule file] explains why a convention exists and when it
applies. A #emph[template] provides a concrete starting point: something to copy
and fill in. A #emph[skill] describes how to carry out a specific procedure.
Together, these three file types cover the full spectrum from motivation to
execution, giving the agent (or the new team member) everything needed to work
within the project's norms.

Rather than loading every rule and skill upfront, the system relies on a short
manifest that says where to start and points to the relevant rules.
#emph[Progressive disclosure] loads a skill only when a task actually needs
it. This keeps the agent's context lean: instead of ingesting thousands of lines
of guidance before writing a single line of code, it pulls in just the skill
that matches the current job, along with the rules that skill depends on.

A well-written skill is a short procedure that points to the rules it follows,
declares its inputs explicitly, and verifies its own work. This structure makes
skills composable and auditable: you can read one in isolation, trace it back to
the rules it implements, and confirm that it checks its output before declaring
success.

Skills are not static, however. They drift whenever the files they reference
change: a rule gets updated, a template gains a new required field, or a naming
convention shifts. Maintaining skills therefore means periodically checking
their references against the current state of the repository and turning any
manual fixes discovered during use back into updated rules. If an agent keeps
making the same correction that no rule covers, that correction belongs in a
rule file, not in the agent's memory.

Finally, skills should be tested like code, albeit with a tolerance for the
nondeterminism inherent in language-model output. Effective testing combines
sample runs against known inputs, golden outputs that capture expected
structure, invariants that must hold regardless of phrasing, and an LLM judge
that evaluates whether the output meets the skill's stated goals. This layered
approach catches both hard failures (a missing section, a broken reference) and
softer regressions (a change in tone or coverage) without demanding exact string
matching.

// From: book_AI_Software_Engineering/lectures_source/Lesson02.01_Skills_Organization_and_Management.smd:1246 '* References'
// Slide: References
#strong[References]

#set text(size: 0.75em)
#references("/book_AI_Software_Engineering/lectures_source/refs.bib")
