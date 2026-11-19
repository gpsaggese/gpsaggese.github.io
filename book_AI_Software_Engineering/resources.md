# Topics
- Rules and AI skills
- Automating PR creation
- Automating PR review
- Automating Documentation
- Automating code coverage
- Automating build
- Automating releasing containers
- Automating unit testing
- Automating code linting for code quality and standards
- Automating code modularization
- Single vs multiple repo
- Dockerization and Dockerized Executables

# Managing Code
- Coding while you sleep
  https://code.claude.com/docs/en/remote-control
  https://code.claude.com/docs/en/web-quickstart
  https://code.claude.com/docs/en/desktop-quickstart
  https://code.claude.com/docs/en/security-guidance
  /install-github-app

# /Users/saggese/src/umd_classes1/helpers_root/papers/AIgentic_Development_System

## Code Quality & Standards

- Linter Framework
  - Architecture & modifying vs non-modifying actions
  - Comprehensive linting rules
  - Integration & extensibility

- Import Cycle Detection
  - Dependency analysis tools
  - Integration into dev workflow

- Package Structure & Import Conventions
  - Goals & rules for imports
  - Package hierarchy & cycle prevention
  - Anatomy of a package
  - Enforcement mechanisms

- Pre-Commit Hook System
  - Enforced checks (branch, author, secrets, file size)
  - Commit message enhancement
  - Installation & configuration

- Code Coverage Tracking & Enforcement
  - Structured coverage by test category
  - CI integration & workflow behavior
  - Enforced thresholds & quality gates
  - Visibility & developer experience
  - Best practices & operational consistency
    - Always upload coverage, even when tests fail
    - Monitor coverage deltas in PRs
    - Global config with repo-specific overrides
    - Weekly review of coverage trends

## Unit Testing

- Testing Philosophy & Motivation
  - Tests as executable specifications
  - First-class concern in development

- hunitest.TestCase Framework
  - Enhanced assertions
  - Golden file testing (check_string)
  - Directory management
  - Text processing utilities
  - Notebook execution support

- Basic Test Structure
  - Three-section pattern
  - Setup, execution, verification

- Golden File Testing
  - How check_string works
  - When to use check_string vs assert_equal
  - Fuzzy matching & text processing

- Test Organization & Conventions
  - File & directory structure
  - Naming conventions
  - Helper methods & DRY principle

- Test Categorization & Execution
  - Fast, slow, superslow test categories
  - Running tests with invoke
  - Timeout & retry behavior

- Mocking Philosophy
  - Mock only external dependencies
  - Do not mock internal code
  - Shared mock setup patterns

- Test Coverage Measurement & Enforcement
  - Running coverage analysis
  - CI/CD integration
  - Interpreting coverage metrics

- Testing Jupyter Notebooks
  - Notebook execution framework
  - Test organization
  - Debugging failing notebook tests

## Coding Architecture

- Runnable Directories Architecture
  - Problem: monorepo vs multi-repo tradeoffs
    - [Mono vs. multi-repo debate (GitKraken, 2023)](https://free.gitkraken.com/hubfs/Mono_v_Multi-Repo_debate_2023.pdf)
    - [Why Google stores billions of lines of code in a single repository](https://dl.acm.org/doi/10.1145/2854146)
    - [What it is like to work in Meta's (Facebook's) monorepo](https://blog.3d-logic.com/2024/09/02/what-it-is-like-to-work-in-metas-facebooks-monorepo/)
    - [Microsoft: How "Mono-repo" and "One Infra" help deliver a better developer experience](https://devblogs.microsoft.com/appcenter/how-mono-repo-and-one-infra-help-us-deliver-a-better-developer-experience/)
    - [Uber: Faster Together, Uber Engineering's iOS Monorepo](https://www.uber.com/blog/ios-monorepo/)
  - Solution: hybrid runnable directory approach
  - Design goals & functionalities
  - Independent yet cohesive modules

- Runnable Directory Design Goals
  - Development functionalities
    - Compose code with a Git submodule (`//helpers`)
    - Add the toolchain to a new project by adding the submodule
    - Complex workflows with Python `invoke`
    - Keep common files in sync across repos (links, automatic diff)
    - Local and remote development in IDEs (PyCharm, VS Code)
  - Python package management
    - Dependency control with [poetry](https://python-poetry.org/) and
      virtual environments
    - Code and container versions in sync (Git, `changelog.txt`)
  - DevOps functionalities
    - Container stages: test/local, dev, prod
    - Standard build, test, retrieve, deploy of containers
    - Alignment of dev environment, deployment, and CI/CD
    - Multi-architecture builds (x86, Arm) on Linux, macOS, WSL
    - Separation of roles: devops builds containers, developers use them
    - Multi-container applications

- Development System Components
  - Docker containerization
  - Thin environment
  - Helpers submodule
  - Git hooks enforcement
  - Recursive test execution

- Containerized Development Workflows
  - Development containers & images
  - Local, dev, prod stages
  - Docker-in-Docker vs sibling containers
  - Multi-container applications

- Thin Environment
  - Minimal dependency setup
  - Shared across runnable directories
  - Bootstrap & development consistency

- Helpers Submodule
  - Centralized toolchain & utilities
  - Common files via symlinks
  - Git hooks management
  - Test infrastructure

## Automation & Workflows

- Invoke Workflows
  - Centralized task registry
  - Python invoke for common operations
  - Development command automation

- Code Repo Automation
  - Standardized label infrastructure
  - Template-based project synchronization
  - Declarative repository settings
  - GitHub metadata management
  - Design principles & workflow behavior
    - Declarative configuration (YAML in version control as source of truth)
    - Reproducible execution (sync runs in Docker containers)
    - Non-destructive by default (explicit confirmation, backup for rollback)
    - Extensibility (future GitHub API features, e.g., views and layouts)
    - Interactive and automated (CI/CD) execution

- CI/CD Automation & Buildmeister Role
  - Build health monitoring
  - Buildmeister dashboard & responsibilities
  - Build break triage & escalation
  - Allure test reporting
  - Post-mortem logging
  - Handover & daily reporting
    - Daily status email: build status, failing tests and owners, issue links,
      expected resolution, dashboard screenshot
    - Rotation handover confirmed by reply from incoming Buildmeister
  - Workflow in practice
    - Slack alert from GitHub Actions bot (#build-notifications)
    - Triage: rerun tests, blame commits, file structured GitHub issue
    - Escalate after 1 hour: disable test with owner consent or revert change

- Docker Container Release Flow
  - Development & production image workflows
  - Version management via changelog.txt
  - Task definition management with ECS
  - Preproduction & production releases
  - Airflow DAG release process
  - Feature release communication
  - Quality gates & automated testing
  - Rollback capabilities
  - Release schedule
  - Pre-release internal testing
  - Central release documentation updates
  - Release communication template

## Dockerized Executables

- Architecture & Design
  - Container image structure
  - Wrapper script responsibilities
  - Repository root as anchor point

- Execution Flow
  - Image availability & discovery
  - Path translation & mounting
  - Exit code propagation

- Container Execution Patterns
  - Children containers (Docker-in-Docker)
  - Sibling containers (preferred)
  - Security & efficiency tradeoffs

- Practical Examples
  - Document formatting & conversion
  - Diagram rendering
  - LaTeX compilation
  - LLM-powered transforms

- Benefits & Trade-offs
  - Reproducibility & consistency
  - Rapid onboarding
  - Independent versioning
  - Docker dependency overhead
  - Image storage & startup latency

- Comparison with Language-Specific Solutions
  - [uv](https://docs.astral.sh/uv/) and similar tools: fast version pinning
    and isolation within one language ecosystem
  - Dockerized executables complement, not replace, them
  - Containers fit best for:
    - Tools in languages not used by the main project
    - System utilities with native dependencies
    - Tools needing specific OS packages or libraries
    - Cross-language toolchains

- Implementation Guidelines
  - Creating minimal images
  - Building wrapper scripts
  - Choosing execution patterns
  - Testing thoroughly
  - Documentation

- Integration with Development Workflows
  - Local development invocation
  - Pre-commit hook integration
  - CI/CD pipeline usage
  - Automated task orchestration

## AI-Optimized Development Infrastructure

- Motivation
  - Same principles help humans and AI agents: discoverability, automation,
    standardization, immediate feedback
  - Infrastructure built for human efficiency enables AI-assisted development

- Self-Documenting Executable Workflows
  - Invoke task discoverability
  - Elimination of ambiguity
  - Cross-repository consistency
  - Encoded best practices

- Agent Instruction Manifests
  - CLAUDE.md as machine-consumable manual
  - Architecture & boundaries
  - Canonical commands
  - Reducing agent onboarding overhead

- Machine-Readable Repository Contracts
  - repo_config.yaml file
  - Repository identity encoding
  - Image naming & registries
  - Direct agent queries

- Context-Bounded Editing
  - Runnable directories as prompt boundaries
  - Self-contained directory structure
  - Reducing context hallucinations
  - Explicit dependency management

- Guardrails & Provenance
  - Formatting automation (pre-commit hooks)
  - Containerized execution consistency
  - Secret hygiene gates
  - Version synchronization
  - Evidence-carrying changes

- The Agentic Loop
  - Plan: summarize change & scope
  - Patch: minimal diff respecting boundaries
  - Prove: layered validation gates
  - Summarize: PR-ready explanation

- Standardized Conventions
  - Predictable file organization
  - Systematic naming patterns
  - Uniform test structure
  - Clear architectural boundaries
  - Reduced decision space for AI

- Container-Based Reproducibility
  - Environment isolation
  - Version synchronization
  - Consistent tool behavior
  - Minimal host dependencies
  - Elimination of "works on my machine"

- Golden File Testing for AI
  - Concrete, interpretable feedback
  - Safe iteration cycles
  - Comprehensive regression detection
  - Elimination of false positives

- Layered Quality Gates
  - Automated corrections (modifying linters)
  - Immediate failure detection
  - Graduated feedback cycles
  - Specific, actionable diagnostics

- Emergent AI-Optimization
  - Degrees of freedom reduction
  - Focus on higher-level concerns
  - Division of labor: humans & AI
  - Positive feedback loop with infrastructure
