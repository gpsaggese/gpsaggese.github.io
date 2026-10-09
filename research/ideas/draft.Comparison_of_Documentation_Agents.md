# Comparison of AI Code Documentation Agents

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- **AI Documentation Agents** are systems that automatically generate and maintain
  software documentation with minimal human intervention
  - Documentation types:
    - Docstrings
    - API documentation
    - Architecture guides
    - README files
    - Inline comments
- Agents span from lightweight inline generators to full documentation frameworks
  - Lightweight inline generators
    - E.g., `GitHub Copilot`, `TabNine`
  - Full documentation frameworks
    - E.g., `Sphinx` with LLM extensions, `Mintlify`, `Documate`, `Javadoc` AI
      enhancements
- Key capabilities:
  - Extract intent from code
  - Generate narrative explanations of complex logic
  - Maintain docs in sync with code changes
  - Auto-generate examples and usage patterns
  - Generate architecture diagrams
  - Multiformat output (Markdown, ReStructuredText, HTML, PDF)
- Agents differ in:
  - Documentation accuracy: does the doc match the actual code?
  - Readability and style consistency
  - Coverage completeness: are all public APIs documented?
  - Ability to regenerate docs automatically after code refactoring
- Most tools expose APIs or IDE integrations, making them accessible in standard
  development workflows (GitHub, GitLab, VS Code, IntelliJ)
- Documentation is often overlooked in AI benchmarks despite being critical for
  software maintainability
  - Good metrics for doc quality are subtle and require human judgment
- Project objective: design a controlled empirical study that benchmarks at least
  three AI documentation agents across multiple Python/TypeScript packages of
  varying complexity and domain
  - Select agents from different categories (IDE, specialized docs, framework)
  - Apply each agent to generate full documentation for the same set of packages
  - Systematically compare:
    - Documentation accuracy
    - Readability and consistency
    - Coverage of public APIs
    - Usefulness of examples
    - Maintainability after code changes

## Formalization

### Comparison of Documentation Agents

| Type             | Name                | Description                                                                 | Website                                             | Strength                    |
| ---------------- | ------------------- | --------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------- |
| IDE-integrated   | GitHub Copilot      | AI-powered docstring and comment generation within IDEs and GitHub          | https://github.com/features/copilot                 | Inline, seamless            |
| Inline generator | TabNine             | Neural code completion with docstring suggestions for multiple languages    | https://www.tabnine.com                             | Language-agnostic           |
| Documentation    | Mintlify            | AI generates and formats documentation from code, maintains docs on website | https://www.mintlify.com                            | Auto-publishing             |
| Sphinx extension | sphinx-autodoc-lllm | Sphinx plugin that uses LLM to enhance auto-generated API documentation     | https://github.com/esbullington/sphinx-autodoc-lllm | Integration with docs build |
| Full platform    | Documate            | AI-powered documentation generation and management for code repositories    | https://www.documate.io                             | Full lifecycle management   |
| Custom           | OpenAI Codex + LLM  | Fine-tuned LLM specifically trained on documentation patterns               | https://platform.openai.com                         | Highly flexible             |

### Documentation Agent Capabilities

| Level                    | Capability                    | Example behaviors                                |
| ------------------------ | ----------------------------- | ------------------------------------------------ |
| L0 -- Suggest completion | Complete partial docstrings   | Autocomplete: `"""Generate a..."""`              |
| L1 -- Generate docstring | Write single method docstring | `def foo(x): """..."""`                          |
| L2 -- Generate suite     | Document entire class/module  | Generate docstrings for 50+ methods              |
| L3 -- Sync with changes  | Update docs when code changes | Regenerate docstring after parameter rename      |
| L4 -- Autonomous docs    | Generate + publish + maintain | Full API docs on website, auto-updated from code |

### Documentation Evaluation Rubric

- Accuracy (40 points)
  - Docstring accurately describes function behavior: 10 pts
  - Parameter descriptions are correct and complete: 10 pts
  - Return type and value description are accurate: 10 pts
  - Exception/error handling documented: 10 pts
- Readability (25 points)
  - Clear, concise prose with good grammar: 10 pts
  - Consistent style across all docstrings: 10 pts
  - Appropriate use of formatting (code blocks, lists): 5 pts
- Coverage (20 points)
  - All public functions documented: 10 pts
  - All public classes and methods documented: 10 pts
- Examples (10 points)
  - Examples provided for key functions: 5 pts
  - Examples are correct and runnable: 5 pts
- Maintainability (5 points)
  - Docs easily regenerated after code changes: 5 pts

## Key Examples

- **Library-level run**: apply each agent to the same Python library and rate
  every generated docstring on accuracy, readability, completeness, and
  maintainability
- **Sync with changes**: rename a function parameter and check whether each
  agent updates the docstring and the usage examples (capability level L3)
- **Hallucinated documentation**: an agent documents that a function raises
  `ValueError` when it never does, or writes an example that does not run;
  executable-example tests catch this while a fluency rating does not

## Questions

1. _Which agents generate the most accurate, readable, complete, and maintainable
   documentation, and under what conditions?_
2. How reliable are the quality ratings themselves? Low inter-rater agreement
   (e.g., Fleiss' Kappa) between reviewers would show that the rubric cannot
   rank the agents.
3. If the best agent's documentation passes the executable-example tests and
   is rated on par with human-written documentation, does the developer's
   job shift from writing documentation to reviewing it?

## Research Topics

- **Multi-Language Comparison**: compare how each agent handles documentation
  across Python, TypeScript, and Go
  - Evaluate language-specific strengths
- **Domain-Specific Documentation**: test agents on domain-specific packages
  - E.g., cryptography, NLP, robotics
  - Measure whether specialized knowledge is reflected in generated docs
- **Interactive Documentation**: evaluate agents' ability to generate interactive
  docs
  - E.g., Jupyter notebooks, animated examples, interactive diagrams
- **Localization**: test whether agents can generate documentation in multiple
  languages
  - Evaluate translation quality
- **Semantic Analysis**: use code similarity tools to measure whether generated
  examples are diverse
  - Check that examples are not just variations of the same pattern
- **Automated Doc Validation**: create tests that verify examples in generated
  documentation actually run and pass
  - Measure correctness
- **User Feedback Integration**: deploy generated documentation to real users
  - Collect feedback on helpfulness
  - Identify which agents produce the most useful docs
- **Cost Analysis**: for cloud-based agents, estimate total cost to document a
  1000-function library
  - Compare to manual documentation effort

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: set up agents and select packages
  - **Agent Setup and Configuration**: install and configure at least three chosen
    documentation agents in isolated environments
    - E.g., GitHub Copilot, Mintlify, Documate
    - Document API keys
    - Document cost (if cloud-based)
    - Document integration with version control and build systems
  - **Package Selection and Baseline**: select 3-5 open-source Python/TypeScript
    packages of varying complexity
    - Simple utility
    - Data science library
    - Web framework
    - CLI tool
  - Strip existing documentation and keep only code
  - Measure baseline:
    - Number of public functions/classes
    - API complexity
    - Code volume
  - Package selection suggestions:
    - **Utility Library** (simple, high coverage)
      - `httpx`: HTTP client library
      - `pendulum`: DateTime manipulation
      - `Click`: CLI framework
      - Source: https://github.com/encode/httpx
    - **Data Science Library** (complex, many functions)
      - `Polars`: DataFrame library
      - `scikit-learn`: ML library (focus on one module)
      - Source: https://github.com/pola-rs/polars
    - **Web Framework** (medium complexity, many classes)
      - `FastAPI`: Web framework
      - `Django REST Framework`: API framework
      - Source: https://github.com/tiangolo/fastapi
    - **CLI Tool** (mixed complexity)
      - `Invoke`: Python task runner
      - `Poetry`: Dependency management
      - Source: https://github.com/pyinvoke/invoke

- Milestone 2: generate documentation
  - **Documentation Generation**: for each agent, generate complete documentation
    - Docstrings for all public APIs
    - Module-level docs
    - README
    - Usage examples
  - Measure:
    - Time to generate
    - Number of tokens/API calls
    - Generation cost

- Milestone 3: assess the generated documentation
  - **Accuracy Assessment**: have experienced developers manually review generated
    documentation and score accuracy on:
    - Docstring matches function behavior
    - Parameter descriptions are correct
    - Return types and examples are accurate
    - Edge cases and exceptions are documented
  - **Readability and Style Consistency**: assess generated documentation on:
    - Prose clarity and grammar
    - Consistency of style across docstrings
    - Appropriate level of detail (not too verbose, not too terse)
    - Presence of headers and structural formatting
  - **Coverage Completeness**: measure what percentage of public APIs are documented
    by each agent
    - Identify which types of APIs are most often missed
    - E.g., decorators, abstract methods, private helpers mistakenly exposed
  - **Example Quality**: evaluate auto-generated examples on:
    - Correctness: do examples run without errors?
    - Clarity: do they illustrate key use cases?
    - Completeness: do they cover common workflows?

- Milestone 4: simulate maintenance and build the scorecard
  - **Maintenance Simulation**: introduce code changes
    - Rename function
    - Add parameter
    - Change return type
    - Refactor module structure
  - Measure:
    - How well each agent regenerates updated documentation
    - Which agent detects deprecated APIs
    - How well agents handle breaking changes
  - **Comparative Scorecard**: build a rubric weighing accuracy, readability,
    coverage, example quality, and maintainability
    - Score each agent
    - Identify which agent excels in each dimension

## References

- **Documentation Benchmarks**:
  - DocString Parser: https://github.com/rr-/docstring_parser
  - PyDocStyle: https://www.pydocstyle.org (PEP 257 checker)
  - Sphinx: https://www.sphinx-doc.org (Python docs generation)
- **Package Sources**:
  - GitHub API: search for repositories by stars, language, topic
  - PyPI: https://pypi.org (Python packages)
  - NPM: https://www.npmjs.com (JavaScript packages)
- **Documentation Quality Metrics**:
  - Flesch Reading Ease: measure readability
  - BLEU Score: evaluate documentation similarity to reference
  - Tree Sitter: parse code structure for coverage analysis
- **Agent Resources**:
  - GitHub Copilot API: https://docs.github.com/en/copilot/quickstart
  - Mintlify Documentation: https://mintlify.com/docs
  - Tabnine API: https://www.tabnine.com/enterprise
- **Human Evaluation**:
  - Likert Scale: standardized rating system for documentation quality
  - Inter-Rater Reliability: calculate Fleiss' Kappa for multiple reviewers
