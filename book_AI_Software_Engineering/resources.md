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
  - Solution: hybrid runnable directory approach
  - Design goals & functionalities
  - Independent yet cohesive modules

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

- CI/CD Automation & Buildmeister Role
  - Build health monitoring
  - Buildmeister dashboard & responsibilities
  - Build break triage & escalation
  - Allure test reporting
  - Post-mortem logging

- Docker Container Release Flow
  - Development & production image workflows
  - Version management via changelog.txt
  - Task definition management with ECS
  - Preproduction & production releases
  - Airflow DAG release process
  - Feature release communication
  - Quality gates & automated testing
  - Rollback capabilities

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
