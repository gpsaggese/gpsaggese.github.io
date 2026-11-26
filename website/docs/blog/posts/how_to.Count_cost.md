---
title: "How to Count LLM Usage Costs"
draft: false
authors:
    - gpsaggese
date: 2026-05-22
description: Tools for tracking LLM API usage costs
categories:
    - LLM
    - Developer Tools
---

TL;DR Track your LLM API spending with open-source tools like `ccusage` and
`OpenUsage` to gain visibility into daily costs, model-level breakdowns, and usage
patterns.

<!-- more -->

## Overview

- LLM-powered coding assistants like Claude Code, GitHub Copilot, and Codex CLI
  boost productivity but API usage costs can add up quickly, especially on
  complex tasks consuming hundreds of thousands of tokens per session

- You need to keep visibility into:
    - How much you spend per day, week, or month
    - Which models drive the costs
    - Whether your usage patterns are efficient

- Several open-source tools exist to help monitor and analyze LLM usage

- In this post we explore how to track costs for an individual running
  requests to provider APIs and OpenRouter, using two practical tools:
    - `ccusage`: CLI tool for generating usage reports across coding agent
      CLIs
    - `OpenUsage`: Terminal-based dashboard for monitoring AI tool usage and
      spend

## Comparison of LLM Cost Computation Methods

- The table below summarizes available methods for computing LLM costs across
  different providers:

| Method                                | OpenAI Direct    | Anthropic Direct  | OpenRouter       | Accuracy  | Implementation Effort | Notes                                    |
| :------------------------------------ | :--------------- | :---------------- | :--------------- | :-------- | :-------------------- | :--------------------------------------- |
| Provider-reported cost                | Yes              | Limited           | Yes              | Very High | Low                   | Best source when available               |
| Response usage fields + pricing table | Yes              | Yes               | Yes              | High      | Low                   | Most common approach                     |
| Cost returned in API response         | No               | No                | Sometimes        | Very High | Very Low              | Depends on provider or gateway           |
| LiteLLM cost calculator               | Yes              | Yes               | Yes              | High      | Low                   | Centralized multi-provider accounting    |
| Langfuse cost tracking                | Yes              | Yes               | Yes              | High      | Medium                | Production observability and attribution |
| Helicone cost tracking                | Yes              | Yes               | Yes              | High      | Medium                | Request-level analytics                  |
| OpenRouter Analytics                  | No               | No                | Yes              | Very High | Very Low              | Includes provider-specific costs         |
| CCUsage                               | No               | Yes (Claude Code) | Partial          | Medium    | Very Low              | Developer-focused usage tracking         |
| OpenCost (Kubernetes)                 | No               | No                | No               | N/A       | Medium                | Measures infrastructure, not token costs |
| GPU runtime accounting                | Self-hosted only | Self-hosted only  | Self-hosted only | High      | Medium                | For vLLM, TGI, SGLang, etc.              |

// TODO(ai_gp): Add a short explanation of the different methods and what the
// columns mean

## Typical Cost Formula

- For direct API providers (OpenAI, Anthropic), the standard formula is:

    ```text
    Cost = (input_tokens * input_price_per_million / 1,000,000)
           + (output_tokens * output_price_per_million / 1,000,000)
    ```

- Example using GPT-4o pricing (as of 2026-06-12):
    - Input: 100,000 tokens at $2.50 / 1M = $0.25
    - Output: 50,000 tokens at $10.00 / 1M = $0.50
    - Total = $0.75

- Key factors to consider:
    - Input and output tokens are priced differently (output is typically 2-4x
      more expensive)
    - Cache hit tokens are billed at a discount (usually 50-90% off input price)
    - Some providers charge for reasoning tokens separately

## OpenRouter Cost Computation

- OpenRouter is a gateway that routes requests to multiple model providers.

- Costs can be computed in two ways:
    1. **OpenRouter Analytics**
        - Use the cost that OpenRouter reports in its Analytics dashboard
          // TODO(ai_gp): Add link
        - Advantages:
            - Reflects the actual routed provider and its specific pricing
            - Accounts for cached token discounts automatically
            - Includes any provider-specific adjustments

    2. **Response Usage Data**
        - Parse the `usage` field from each API response:
   
            ```json
            {
                "usage": {
                    "prompt_tokens": 1234,
                    "completion_tokens": 567
                }
            }
            ```
   
        - Then multiply token counts against the OpenRouter model pricing catalog:
   
            ```text
            Cost = (prompt_tokens * model_input_price)
                   + (completion_tokens * model_output_price)
            ```

### Viewing OpenRouter Logs

// TODO(ai_gp): Add links to the pages if possible

- OpenRouter provides several ways to inspect usage and costs:

- **Activity page** (Usage Logs):
    - Navigate to OpenRouter dashboard -> **Activity**
    - View requests, models, providers, API key activity, token usage, and costs

- **Full prompt and response logging**:
    - Go to **Settings** -> **Observability** in the dashboard
    - Enable **Input and Output Logging**
    - Open the **Logs** page to inspect prompts, completions, model, provider,
      token counts, and request cost
    - Note: logging only applies to requests made after the feature is enabled

- **Programmatic retrieval**:
    - Analytics endpoints require a **Management Key** (not a standard API key):

        ```bash
        > curl https://openrouter.ai/api/v1/analytics/activity \
            -H "Authorization: Bearer YOUR_MANAGEMENT_KEY"
        ```

    - Available filters include date range, API key, user, endpoint, and model

- **Per-request tracking**:
    - Every OpenRouter response includes usage information with prompt tokens,
      completion tokens, total tokens, and cost when available
    - Example in Python:

        ```python
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )

        print(response.usage)
        ```

- Useful dashboard pages:
    - **Activity**: Usage history
    - **Logs**: Prompts and completions
    - **Analytics**: Cost and token breakdown
    - **Settings -> Observability**: Input and Output Logging toggle

## `ccusage`: CLI Usage Reports

### What It Does

- [ccusage](https://ccusage.com) is an open-source CLI tool that:
    - Reads local usage logs from supported coding agent CLIs
    - Estimates USD spend from token counts and model pricing
    - Generates reports across daily, weekly, monthly, and session timeframes
    - Tracks cache creation and cache read tokens separately
    - Exports data in structured JSON format for programmatic use

// TODO(ai_gp): Improve this list by splitting in better category.

- It supports a wide range of coding assistants, including:
    - Claude Code, Codex CLI, OpenCode
    - Amp, Droid, Codebuff
    - Hermes Agent, pi-agent, Goose
    - GitHub Copilot CLI, Gemini CLI, Qwen, Kilo

- A key design principle is privacy: all data reads from local logs, and nothing
  uploads to external servers

### Installation

- Install globally via npm:

    ```bash
    > npm install -g ccusage
    ```

- Alternatively, run without installation using `bunx`, `pnpm dlx`, or `npx`
- Requires Bun 1.3+ (recommended) or Node.js as a runtime

### Basic Usage

- Generate a daily usage report with cost breakdown:

    ```bash
    > ccusage daily --breakdown --no-color
    ```

- The output is a terminal table showing date, models used, input and output
  token counts, and estimated cost in USD

- Example output (simplified):

    ```text
    ┌──────────┬──────────────────────┬────────────┬──────────┬─────────────┐
    │ Date     │ Models                │      Input │   Output │  Cost (USD) │
    ├──────────┼──────────────────────┼────────────┼──────────┼─────────────┤
    │ 2026-06-10│ - 4.5-haiku          │  8,790,570 │  123,784 │       $4.12 │
    │          │ - 4.6-sonnet          │            │          │             │
    │          │ - deepseek-v4-flash   │            │          │             │
    ├──────────┼──────────────────────┼────────────┼──────────┼─────────────┤
    │ Total    │                      │  8,790,570 │  123,784 │       $4.12 │
    └──────────┴──────────────────────┴────────────┴──────────┴─────────────┘
    ```

// TODO(ai_gp): Create more command examples and output example

- Additional commands and flags:
    - `ccusage weekly` and `ccusage monthly` for longer timeframes
    - `--breakdown` flag for per-model cost breakdowns
    - `--json` flag for structured data export
    - `--since` flag to specify a start date
    - Offline mode using pre-cached pricing data

- `ccusage` offers a `blocks --live` command for real-time monitoring of Claude
  Code sessions
- This is useful for keeping an eye on costs during active development sessions

## OpenUsage: Terminal Dashboard

### What It Does

- [OpenUsage](https://github.com/janekbaraniewski/openusage) is an open-source,
  terminal-first dashboard for monitoring spend across AI coding tools and API
  platforms

- Unlike ccusage, which focuses on CLI reports, OpenUsage provides a full
  terminal user interface (TUI) with real-time updates

- It auto-detects installed AI tools and API keys on your workstation and shows
  live quota, usage, spend, resets, and rate limits

### Installation

- Install via npm:

    ```bash
    > npm install -g openusage
    ```

- It runs entirely locally with zero configuration required

### Key Features

- **Live TUI dashboard**: Real-time display of spend, quotas, rate limits,
  tokens, and per-model breakdowns
    - Supports 17 built-in themes with custom theme file support

- **Auto-detection**: Automatically detects installed AI tools and environment
  variables containing API keys

- **Background daemon**: Continuously collects data into a local SQLite database
  for historical tracking

- **Headless reports**: CLI commands for `daily`, `weekly`, `monthly`,
  `session`, and `blocks` reports in table or JSON format

- **Claude Code integration**:
    - Statusline support showing session cost, burn rate, and context window
      usage
    - Tmux status bar integration with provider-specific logos

- **Export capabilities**: Export to JSON or CSV, plus Prometheus metrics
  support

- **Supported providers** (34 total):
    - Coding agents and IDEs: Claude Code, Cursor, GitHub Copilot, Codex CLI,
      Gemini CLI, OpenCode, Ollama
    - API platforms: OpenAI, Anthropic, OpenRouter, Groq, Mistral, DeepSeek,
      Grok, Perplexity, Google Gemini API, and many more

### Use Cases

- OpenUsage is ideal for developers who:
    - Want a real-time dashboard they can leave running in a terminal window
    - Use multiple AI tools and want a unified view of all spending
    - Need to track usage across both coding agents and direct API access
    - Want to integrate cost metrics into existing monitoring infrastructure via
      Prometheus

## Choosing Between ccusage and OpenUsage

- Both tools serve the same general purpose but have different strengths:

| Aspect               | ccusage                 | OpenUsage                       |
| :------------------- | :---------------------- | :------------------------------ |
| Primary interface    | CLI reports             | TUI dashboard + CLI reports     |
| Real-time monitoring | `blocks --live` command | Live dashboard with daemon      |
| Model coverage       | Coding assistants focus | 34 providers, including APIs    |
| Data export          | JSON                    | JSON, CSV, Prometheus           |
| Auto-detection       | Manual source selection | Auto-detects tools and API keys |
| Historical tracking  | Per-report queries      | SQLite-backed daemon            |
| Theme support        | Terminal tables         | 17 built-in themes + custom     |

- **Use ccusage if** you need quick, one-off reports and primarily use coding
  assistant CLIs

- **Use OpenUsage if** you want a persistent dashboard, use multiple AI tools,
  and need real-time visibility into costs

- I use both since they complement each other well

## Practical Tips for Managing LLM Costs

- Track costs regularly:
    - Run `ccusage daily` at the end of each day
    - Keep OpenUsage running in the background for continuous awareness

- Understand which models drive your costs:
    - Use the `--breakdown` flag to see per-model costs
    - Expensive frontier models (Opus, Sonnet) can dominate spend even with
      minimal usage

- Monitor cache efficiency:
    - Both tools track cache creation and cache read tokens
    - A high cache hit rate reduces costs significantly
    - Claude Code automatically caches prompts, so reusing context across
      sessions helps

- Set mental budgets:
    - Once you know your typical daily spend, you can spot anomalies quickly
    - A sudden spike often indicates an inefficient workflow or a runaway agent
      loop

# To reorg

> more .claude/cc_usage.sh
#!/usr/bin/env bash
ccusage claude daily --breakdown --since $(date +%Y-%m-%d) --compact --no-color

⠦ Loading usage logs (0/1) :: Claude

╭────────────────────────────────────────────╮
│                                            │
│   Claude Code Token Usage Report - Daily   │
│                                            │
╰────────────────────────────────────────────╯

┌─────────────────────────────────┬────────────────────────────────────────────────────────────────────────┬─────────────┬───────────┬─────────────┐
│ Date                            │ Models                                                                 │       Input │    Output │  Cost (USD) │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│ 2026-06-17                      │ - 4.5-haiku                                                            │  11,795,060 │    92,632 │       $2.54 │
│                                 │ - deepseek/deepseek-v4-flash                                           │             │           │             │
│                                 │ - haiku-4-5                                                            │             │           │             │
│                                 │ - openai/gpt-oss-20b                                                   │             │           │             │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│   └─ deepseek/deepseek-v4-flash │                                                                        │  11,769,956 │    85,233 │       $2.27 │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│   └─ 4.5-haiku                  │                                                                        │          35 │     5,437 │       $0.17 │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│   └─ haiku-4-5                  │                                                                        │          66 │     1,840 │       $0.10 │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│   └─ openai/gpt-oss-20b         │                                                                        │      25,003 │       122 │       $0.00 │
├─────────────────────────────────┼────────────────────────────────────────────────────────────────────────┼─────────────┼───────────┼─────────────┤
│ Total                           │                                                                        │  11,795,060 │    92,632 │       $2.54 │
└─────────────────────────────────┴────────────────────────────────────────────────────────────────────────┴─────────────┴───────────┴─────────────┘

Running in Compact Mode
Expand terminal width to see cache metrics and total tokens

haiku-4-5 is the ANthropic plan and 4.5-haiku is the Openrouter version

##

There is no API call for personal plans
