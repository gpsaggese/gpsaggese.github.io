---
title: "Pi.dev vs Claude Code: AI Coding Agent"
draft: true
authors:
  - gpsaggese
date: 2026-05-23
categories:
  - AI Tools
  - Developer Tools
---

TL;DR: Claude Code offers integrated coding workflows across several
interfaces, while pi.dev offers a customizable terminal agent with a choice of
model providers.

<!-- more -->

## Introduction

Both **Claude Code** and **pi.dev** are AI coding agents: tools that use a large
language model (LLM) to inspect code, edit files, and run development commands.
Both can help with debugging, refactoring, and tests, but they emphasize
different workflows.

**Claude Code** is Anthropic's commercial coding agent. It runs through a
command-line interface (CLI) in your terminal, integrations with an integrated
development environment (IDE) such as VS Code or JetBrains, the Claude desktop
app, and a browser. Features and account requirements vary by interface; see
the [official overview](https://code.claude.com/docs/en/overview).

**pi.dev** is an open-source, terminal-first agent created by Mario Zechner.
You configure its model connection and can extend its behavior with TypeScript.
The agent runs locally, but inference, the process of generating model
responses, happens wherever your chosen model runs. A cloud connection sends
context to that provider; a local model server processes requests on your
infrastructure.

This is a **documentation-based comparison, not a hands-on benchmark**. The four
scenarios illustrate possible workflows rather than recorded results. The
technical descriptions follow the linked official documentation checked on
October 23, 2026.

### When to Use Claude Code

- You want terminal, IDE, desktop, or browser access
- You value supported Git and GitHub integrations
- You want configurable permission controls and visual diff review

### When to Use Pi.dev

- You want to choose model providers or configure local inference
- You need custom tools or changes to the agent's behavior
- You prefer MIT-licensed, open-source tooling

Both tools can make incorrect changes even when tests pass. Review the actual
diff, the lines changed in your files, and run checks appropriate to the task.
Neither tool alone establishes regulatory compliance or guarantees privacy;
those depend on your deployment and connected services.

### Related Tools

- **GitHub Copilot**: Offers completion and
  [agent workflows](https://github.blog/news-insights/product-news/github-copilot-meet-the-new-coding-agent/)
- **Continue.dev**: Provides IDE assistance with chat, planning, and
  [agent modes](https://docs.continue.dev/ide-extensions/agent/quick-start)
- **OpenClaw**: A broader
  [self-hosted assistant](https://docs.openclaw.ai/help/faq/what-is-openclaw)
- **LangChain**: A framework for
  [building custom agents](https://docs.langchain.com/oss/python/langchain/overview)

## Prerequisites

### For Claude Code

- A supported Claude subscription, Claude Console account with API credits, or
  supported cloud-provider access for the CLI
- The terminal, editor, desktop app, or browser required by your chosen
  interface
- Internet connectivity for the hosted model connection

An application programming interface (API) lets software exchange requests and
responses. API credits pay for model requests; a subscription is a separate
access option. The native CLI installer doesn't require Node.js. Check the
[quickstart](https://code.claude.com/docs/en/quickstart) and
[setup guide](https://code.claude.com/docs/en/setup) for platform requirements.

### For Pi.dev

- Node.js 22.19+ for the npm installation below; npm is Node.js's package
  manager
- Model access through a supported subscription, API key, or local server
- Connectivity for installation, model downloads, and cloud providers

An API key authenticates requests to a provider. Offline use requires local
models and dependencies already installed, plus tools configured to avoid
external services.

## Installation and Setup

### Claude Code Installation

For macOS, Linux, or Windows Subsystem for Linux (WSL), use the native
installer:

```bash
> curl -fsSL https://claude.ai/install.sh | bash
```

Open a new terminal, verify installation, and start in your project:

```bash
> claude --version
> cd my-project
> claude
```

Follow the first-use login prompts. Alternatively, on macOS:

```bash
> brew install --cask claude-code
```

For browser access, open [claude.ai/code](https://claude.ai/code) with an eligible
account. IDE users install the VS Code extension or JetBrains plugin; JetBrains
also requires the CLI. Desktop users open the Code tab in the Claude app. The
[overview](https://code.claude.com/docs/en/overview) links to interface-specific
instructions, including native Windows setup.

### Pi.dev Installation

Install and verify the coding agent:

```bash
> npm install -g --ignore-scripts @earendil-works/pi-coding-agent
> pi --version
> cd my-project
> pi
```

Inside Pi, sign in and select a model:

```text
prompt> /login
prompt> /model
```

Alternatively, use a supported provider's environment variable:

```bash
> export OPENAI_API_KEY="your-openai-api-key"
> pi --provider openai
```

For Anthropic, use `ANTHROPIC_API_KEY` and `--provider anthropic` instead.
Pi's [quickstart](https://pi.dev/docs/latest/quickstart) documents other install
methods; its [model guide](https://pi.dev/docs/latest/models) covers connections.

## Core Concepts

### Claude Code's Model

Claude Code combines file editing and shell commands with configurable
permissions. A shell executes terminal commands; permission rules determine
which actions need approval. Sandboxing separately restricts access to resources
such as files and network services. See the
[permission guide](https://code.claude.com/docs/en/permissions).

- **Interfaces**: Review changes in the terminal or supported visual interfaces
- **Parallel work**: Delegate tasks to
  [subagents](https://code.claude.com/docs/en/sub-agents), additional agent
  sessions that handle parts of a task
- **Models**: Aliases such as `sonnet`, `opus`, and `haiku` select model
  options; availability depends on your account and deployment; see the
  [model guide](https://code.claude.com/docs/en/model-config)

### Pi.dev's Model

Pi starts with `read`, `write`, `edit`, and `bash` enabled, with additional
built-in tools available. It saves sessions without requiring an extension.
See the [CLI reference](https://pi.dev/docs/latest/cli).

- **Connections**: Use supported providers or compatible endpoints, the server
  addresses and API formats used for model requests
- **Customization**: TypeScript extensions can register tools and react to
  events
- **Execution**: Tools run with your operating-system permissions; Pi doesn't
  ask before every tool call

Local sessions can contain sensitive material, and connected providers or tools
may receive code and prompts. Read [how Pi works](https://pi.dev/docs/latest/how-pi-works)
and its [security guidance](https://pi.dev/docs/latest/security).

## Illustrative Workflows

### Example 1: Quick Python Project Fix (Claude Code)

Scenario: A small Python project has a failing test.

```bash
> cd my-project
> claude
```

```text
prompt> Debug test_utils.py: expected 10, got 5.
        Investigate the cause, fix it, and run the relevant tests.
```

The agent could read the test and implementation, investigate the mismatch,
edit the code subject to permissions, and present the result. IDE integrations
provide visual diffs; the terminal also supports the coding workflow.

**Pros:** Code exploration, editing, and checks fit one conversation.

**Limitations:** Hosted inference needs connectivity, and usage limits depend
on your account. Pi can perform the same read-edit-test loop; this scenario
establishes no speed or correctness advantage.

### Example 2: Migrating to Self-Hosted Models (pi.dev)

Scenario: A healthcare company wants on-premise inference. Install
[Ollama](https://docs.ollama.com/cli), then start its service in a separate
terminal if it isn't running:

```bash
> ollama serve
```

Download a model before disconnecting:

```bash
> ollama pull qwen2.5-coder:7b
```

Merge this provider into `~/.pi/agent/models.json`:

```json
{
    "providers": {
        "ollama": {
            "baseUrl": "http://localhost:11434/v1",
            "api": "openai-completions",
            "apiKey": "ollama",
            "models": [{ "id": "qwen2.5-coder:7b" }]
        }
    }
}
```

This follows Pi's [endpoint example](https://pi.dev/docs/latest/models), not a
model-quality recommendation. The placeholder key is ignored by local Ollama.
See [configuration](https://pi.dev/docs/latest/configuration) for file locations.

```bash
> cd healthcare-project
> pi --provider ollama --model qwen2.5-coder:7b
```

```text
prompt> Refactor the patient data loader to handle missing fields.
        Run the relevant tests and show the changes.
```

**Pros:** Local inference keeps model requests on your infrastructure.

**Limitations:** You manage hardware, compatibility, and performance. Extensions
and other tools may still contact external services. Network controls and data
handling need separate evaluation before using sensitive information.

### Example 3: Custom Tool Extension (pi.dev)

Scenario: You want an internal code generator available to the agent. This
TypeScript skeleton returns code; replace its generation logic with your
integration. It follows Pi's
[custom-tool example](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/hello.ts).

```typescript
// my_tools/custom_generator.ts
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
    pi.registerTool(
        defineTool({
            name: "generate_boilerplate",
            label: "Generate Boilerplate",
            description: "Return TypeScript boilerplate for a new service",
            parameters: Type.Object({
                service_name: Type.String({
                    description: "Service name as a TypeScript identifier",
                    pattern: "^[A-Za-z_$][A-Za-z0-9_$]*$",
                }),
            }),
            async execute(_toolCallId, params) {
                const code = `export class ${params.service_name}Service {}\n`;
                return {
                    content: [{ type: "text", text: code }],
                    details: { service_name: params.service_name },
                };
            },
        }),
    );
}
```

Load the extension:

```bash
> pi --extension ./my_tools/custom_generator.ts
```

```text
prompt> Generate boilerplate for the payment service.
```

Pi could call the tool, then write its output using a separate file-writing
action. Extension discovery is covered in the
[extension guide](https://pi.dev/docs/latest/extensions).

**With Claude Code:** Expose custom tools through the Model Context Protocol
([MCP](https://code.claude.com/docs/en/mcp)), a standard for connecting agents to
tools and data. Skills package reusable instructions; hooks run automation in
response to events. Plugins distribute integrations. Both agents are extensible,
using different [integration mechanisms](https://code.claude.com/docs/en/features-overview).

### Example 4: Collaborative Debugging (Claude Code)

Scenario: A team needs shared context and version history for a production bug.
Use separate working copies or Git worktrees, directories that let different
branches be checked out simultaneously, for concurrent tasks.

**With Claude Code:**

- Share project instructions through `CLAUDE.md` and reusable skills
- Investigate the bug, run checks, and review the diff
- Commit reviewed changes and share a pull request for team review
- Optionally use the [GitHub Actions integration](https://code.claude.com/docs/en/github-actions)

**With pi.dev:**

- Share instructions through `AGENTS.md` and distribute extensions or skills
- Use Git branches and pull requests for the same team-review process
- Save and resume individual sessions as needed

Git records committed changes; neither agent replaces coordination or review.

## Tips and Gotchas

### Claude Code Tips

- Check permission settings before relying on approval prompts
- Break large refactors into focused tasks with review checkpoints
- Check model availability and usage costs for your deployment

### Pi.dev Tips

- Check model tool-calling support and context capacity, the amount of material
  a model can process in a request
- Review extensions before loading them
- Measure performance on your hardware rather than assuming a cloud comparison

### Common Mistakes

- **Reviewing summaries alone**: Inspect the actual diff and test results
- **Oversized requests**: Split work by module; file count alone doesn't define
  the context limit
- **Missing connections**: Configure authentication or `models.json` before
  selecting the model

## Next Steps

### Learning More

- **Claude Code**: Read [best practices](https://code.claude.com/docs/en/best-practices)
  or take [Claude Code in Action](https://academy.claude.com/courses/claude-code-in-action)
- **Pi.dev**: Explore the [repository](https://github.com/earendil-works/pi) and
  extension documentation

### Hybrid Approach

Try representative tasks in separate projects. Compare correctness, setup
effort, and cost against your own requirements; you don't have to choose one
agent for everything.

### Related Topics

- **Local models**: Ollama, vLLM, and llama.cpp
- **Custom agents**: LangChain and agent integration APIs
- **Prompt caching**: Learn [how Claude Code reuses cached context](https://code.claude.com/docs/en/prompt-caching)

## Conclusion

Claude Code offers multiple interfaces and supported development integrations.
Pi.dev emphasizes provider choice and direct customization of a terminal agent.
Choose based on your deployment, tools, and review process, then evaluate the
results on your own project.
