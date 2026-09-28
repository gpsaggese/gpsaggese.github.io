# Comparison of AI Code Documentation Agents

## Description

- **AI Documentation Agents** are systems that automatically generate and maintain
  software documentation, including docstrings, API documentation, architecture
  guides, README files, and inline comments with minimal human intervention
- They span from lightweight inline generators (e.g., `GitHub Copilot`, `TabNine`) to
  full documentation frameworks (e.g., `Sphinx` with LLM extensions, `Mintlify`,
  `Documate`, `Javadoc` AI enhancements)
- Key capabilities include extracting intent from code, generating narrative
  explanations of complex logic, maintaining docs in sync with code changes,
  auto-generating examples and usage patterns, generating architecture diagrams, and
  multiformat output (Markdown, ReStructuredText, HTML, PDF)
- Agents differ in documentation accuracy (does the doc match the actual code?),
  readability and style consistency, coverage completeness (all public APIs
  documented?), and ability to regenerate docs automatically after code refactoring
- Most tools expose APIs or IDE integrations, making them accessible in standard
  development workflows (GitHub, GitLab, VS Code, IntelliJ)
- This project teaches students that documentation is often overlooked in AI
  benchmarks despite being critical for software maintainability—good metrics for doc
  quality are subtle and require human judgment

## Comparison of Documentation Agents

| Type             | Name                | Description                                                                 | Website                                             | Strength                    |
| ---------------- | ------------------- | --------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------- |
| IDE-integrated   | GitHub Copilot      | AI-powered docstring and comment generation within IDEs and GitHub          | https://github.com/features/copilot                 | Inline, seamless            |
| Inline generator | TabNine             | Neural code completion with docstring suggestions for multiple languages    | https://www.tabnine.com                             | Language-agnostic           |
| Documentation    | Mintlify            | AI generates and formats documentation from code, maintains docs on website | https://www.mintlify.com                            | Auto-publishing             |
| Sphinx extension | sphinx-autodoc-lllm | Sphinx plugin that uses LLM to enhance auto-generated API documentation     | https://github.com/esbullington/sphinx-autodoc-lllm | Integration with docs build |
| Full platform    | Documate            | AI-powered documentation generation and management for code repositories    | https://www.documate.io                             | Full lifecycle management   |
| Custom           | OpenAI Codex + LLM  | Fine-tuned LLM specifically trained on documentation patterns               | https://platform.openai.com                         | Highly flexible             |

## Documentation Agent Capabilities

| Level                    | Capability                    | Example behaviors                                |
| ------------------------ | ----------------------------- | ------------------------------------------------ |
| L0 -- Suggest completion | Complete partial docstrings   | Autocomplete: `"""Generate a..."""`              |
| L1 -- Generate docstring | Write single method docstring | `def foo(x): """..."""`                          |
| L2 -- Generate suite     | Document entire class/module  | Generate docstrings for 50+ methods              |
| L3 -- Sync with changes  | Update docs when code changes | Regenerate docstring after parameter rename      |
| L4 -- Autonomous docs    | Generate + publish + maintain | Full API docs on website, auto-updated from code |

## Project Objective

Design a controlled empirical study that benchmarks at least three AI documentation
agents across multiple Python/TypeScript packages of varying complexity and domain
The project aims to answer: _Which agents generate the most accurate, readable,
complete, and maintainable documentation: and under what conditions?_ Students will
select agents from different categories (IDE, specialized docs, framework), apply
each to generate full documentation for the same set of packages, and systematically
compare: documentation accuracy, readability and consistency, coverage of public
APIs, usefulness of examples, and maintainability after code changes

## Tasks

- **Agent Setup & Configuration**: Install and configure at least three chosen
  documentation agents (e.g., GitHub Copilot, Mintlify, Documate) in isolated
  environments; document API keys, cost (if cloud-based), and integration with
  version control/build systems
- **Package Selection & Baseline**: Select 3–5 open-source Python/TypeScript packages
  of varying complexity (simple utility, data science library, web framework, CLI
  tool); strip existing documentation (keep only code); measure baseline: number of
  public functions/classes, API complexity, code volume
- **Documentation Generation**: For each agent, generate complete documentation:
  docstrings for all public APIs, module-level docs, README, usage examples; measure:
  time to generate, number of tokens/API calls, and generation cost
- **Accuracy Assessment**: Have experienced developers manually review generated
  documentation and score accuracy on: (a) docstring matches function behavior, (b)
  parameter descriptions are correct, (c) return types and examples are accurate, (d)
  edge cases and exceptions are documented
- **Readability & Style Consistency**: Assess generated documentation on: (a) prose
  clarity and grammar, (b) consistency of style across docstrings, (c) appropriate
  level of detail (not too verbose, not too terse), (d) presence of headers and
  structural formatting
- **Coverage Completeness**: Measure what percentage of public APIs are documented by
  each agent; identify which types of APIs are most often missed (decorators,
  abstract methods, private helpers mistakenly exposed)
- **Example Quality**: Evaluate auto-generated examples on: (a) correctness (do
  examples run without errors?), (b) clarity (do they illustrate key use cases?), (c)
  completeness (do they cover common workflows?)
- **Maintenance Simulation**: Introduce code changes (rename function, add parameter,
  change return type, refactor module structure) and measure: (a) how well each agent
  regenerates updated documentation, (b) which agent detects deprecated APIs, (c) how
  well agents handle breaking changes
- **Comparative Scorecard**: Build a rubric weighing accuracy, readability, coverage,
  example quality, and maintainability; score each agent and identify which agent
  excels in each dimension

## Package Selection Suggestions

- **Utility Library** (Simple, high coverage)
  - `httpx`: HTTP client library
  - `pendulum`: DateTime manipulation
  - `Click`: CLI framework
  - Source: https://github.com/encode/httpx
- **Data Science Library** (Complex, many functions)
  - `Polars`: DataFrame library
  - `scikit-learn`: ML library (focus on one module)
  - Source: https://github.com/pola-rs/polars
- **Web Framework** (Medium complexity, many classes)
  - `FastAPI`: Web framework
  - `Django REST Framework`: API framework
  - Source: https://github.com/tiangolo/fastapi
- **CLI Tool** (Mixed complexity)
  - `Invoke`: Python task runner
  - `Poetry`: Dependency management
  - Source: https://github.com/pyinvoke/invoke

## Documentation Evaluation Rubric

### Accuracy (40 Points)

- Docstring accurately describes function behavior: 10 pts
- Parameter descriptions are correct and complete: 10 pts
- Return type and value description are accurate: 10 pts
- Exception/error handling documented: 10 pts

### Readability (25 Points)

- Clear, concise prose with good grammar: 10 pts
- Consistent style across all docstrings: 10 pts
- Appropriate use of formatting (code blocks, lists): 5 pts

### Coverage (20 Points)

- All public functions documented: 10 pts
- All public classes and methods documented: 10 pts

### Examples (10 Points)

- Examples provided for key functions: 5 pts
- Examples are correct and runnable: 5 pts

### Maintainability (5 Points)

- Docs easily regenerated after code changes: 5 pts

## Bonus Ideas

- **Multi-Language Comparison**: Compare how each agent handles documentation across
  Python, TypeScript, and Go; evaluate language-specific strengths
- **Domain-Specific Documentation**: Test agents on domain-specific packages
  (cryptography, NLP, robotics); measure whether specialized knowledge is reflected
  in generated docs
- **Interactive Documentation**: Evaluate agents' ability to generate interactive
  docs (Jupyter notebooks, animated examples, interactive diagrams)
- **Localization**: Test whether agents can generate documentation in multiple
  languages; evaluate translation quality
- **Semantic Analysis**: Use code similarity tools to measure whether generated
  examples are diverse (not just variations of the same pattern)
- **Automated Doc Validation**: Create tests that verify examples in generated
  documentation actually run and pass; measure correctness
- **User Feedback Integration**: Deploy generated documentation to real users;
  collect feedback on helpfulness and identify which agents produce most useful docs
- **Cost Analysis**: For cloud-based agents, estimate total cost to document a
  1000-function library; compare to manual documentation effort

## Useful Resources

- **Documentation Benchmarks**:
  - DocString Parser: https://github.com/rr-/docstring_parser
  - PyDocStyle: https://www.pydocstyle.org (PEP 257 checker)
  - Sphinx: https://www.sphinx-doc.org (Python docs generation)
- **Package Sources**:
  - GitHub API: Search for repositories by stars, language, topic
  - PyPI: https://pypi.org (Python packages)
  - NPM: https://www.npmjs.com (JavaScript packages)
- **Documentation Quality Metrics**:
  - Flesch Reading Ease: Measure readability
  - BLEU Score: Evaluate documentation similarity to reference
  - Tree Sitter: Parse code structure for coverage analysis
- **Agent Resources**:
  - GitHub Copilot API: https://docs.github.com/en/copilot/quickstart
  - Mintlify Documentation: https://mintlify.com/docs
  - Tabnine API: https://www.tabnine.com/enterprise
- **Human Evaluation**:
  - Likert Scale: Standardized rating system for documentation quality
  - Inter-Rater Reliability: Calculate Fleiss' Kappa for multiple reviewers
