---
title: "How to Use Claude Code with OpenRouter"
draft: false
authors:
    - gpsaggese
date: 2026-06-05
categories:
    - AI Tools
    - LLM
---

TL;DR Learn how to configure Claude Code to route LLM calls through OpenRouter,
enabling access to models from multiple providers through a single API.

<!-- more -->

- [OpenRouter](https://openrouter.ai/) is an API gateway that gives you access
  to LLMs from multiple providers (Anthropic, OpenAI, DeepSeek, Meta, Google,
  etc.) through a single endpoint
    - Use Claude Code with non-Anthropic models like DeepSeek or OpenAI GPT
    - Or mix models from different providers depending on the task

- For a general introduction to Claude Code, see the
  [Claude Code setup guide](https://docs.anthropic.com/en/docs/claude-code/setup)

- This guide covers:
    - Setting up OpenRouter, with and without BYOK (Bring Your Own Key)
    - Testing the API connection
    - Configuring Claude Code to route through OpenRouter

## What is OpenRouter and BYOK

- OpenRouter acts as a proxy between your client and LLM providers
    - You send requests to `https://openrouter.ai/api/v1`
    - OpenRouter routes them to the appropriate provider

- **BYOK (Bring Your Own Key)** lets you use your existing API keys from
  providers (Anthropic, OpenAI, etc.) instead of buying credits from OpenRouter
    - Useful if you already have subscriptions or credits with a provider
    - Register your keys with OpenRouter and traffic routes through your own
      accounts

## Step 1: Configure OpenRouter with BYOK

- Generate an OpenRouter API key (`sk-or-v1-...`) for your client applications

- To bring your own keys, create API keys for your providers at:
    - [Anthropic Console](https://platform.claude.com/settings/workspaces/default/keys)
    - [OpenAI API Keys](https://platform.openai.com/api-keys)

    | Provider  | Environment Variable | Key Format         |
    | :-------- | :------------------- | :----------------- |
    | Anthropic | `ANTHROPIC_API_KEY`  | `sk-ant-api03-...` |
    | OpenAI    | `OPENAI_API_KEY`     | `sk-proj-...`      |

    - Note that Anthropic encodes in the key `...-api-03-...` since you need
      to use metered use

- Register your BYOK keys with OpenRouter at:
  [BYOK Settings](https://openrouter.ai/workspaces/default/byok)

## Step 2: Test the OpenRouter API

- List available models through OpenRouter:

    ```bash
    > curl https://openrouter.ai/api/v1/models \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        | jq '.data[].id' | sort
    ```

- Test a simple completion:

    ```bash
    > curl https://openrouter.ai/api/v1/chat/completions \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
          "model": "openai/gpt-5",
          "messages": [
            {
              "role": "user",
              "content": "Reply with the word OK"
            }
          ]
        }'
    ```

## Step 3: Configure Claude Code for OpenRouter

- Claude Code uses Anthropic's SDK, which connects to `api.anthropic.com` by
  default
- To route through OpenRouter, set these environment variables:

    - Send the API requests from Claude Code to OpenRouter using their API and key
        ```bash
        > export ANTHROPIC_BASE_URL=https://openrouter.ai/api
        > export ANTHROPIC_AUTH_TOKEN=$OPENROUTER_API_KEY
        ```
    - Select the models to use for the different Claude Code tier (e.g., using
      OpenAI GPT-5 using OpenRouter)
        ```bash
        > export ANTHROPIC_DEFAULT_HAIKU_MODEL=openai/gpt-5
        > export ANTHROPIC_DEFAULT_OPUS_MODEL=openai/gpt-5
        > export ANTHROPIC_DEFAULT_SONNET_MODEL=openai/gpt-5
        ```
    - Unset the direct Anthropic key to avoid conflicts
        ```
        > unset ANTHROPIC_API_KEY
        ```

- Verify the environment is configured correctly:

    ```bash
    > env | grep ANT | sort
    ANTHROPIC_AUTH_TOKEN=sk-or-v1-...
    ANTHROPIC_BASE_URL=https://openrouter.ai/api
    ANTHROPIC_DEFAULT_HAIKU_MODEL=openai/gpt-5
    ANTHROPIC_DEFAULT_OPUS_MODEL=openai/gpt-5
    ANTHROPIC_DEFAULT_SONNET_MODEL=openai/gpt-5
    ```

### Testing Connection to OpenRouter with the Anthropic Python SDK

<!-- TODO(ai_gp): move to repo and point to it -->

- Run the script `.claude/test_openrouter_api.py` to verify that OpenRouter
  responds when using the Anthropic API

    ```bash
    > .claude/test_openrouter_api.py
    Endpoint : https://openrouter.ai/api
    Model    : openai/gpt-5
    Testing...

    SUCCESS
    ```

## Step 4: Using Different Models

- Once OpenRouter is configured, point Claude Code's model tiers at whatever
  models you want
- For help choosing which model fits your workflow, see
  [How to Compare and Choose LLM Models](how_to.Compare_LLM_models.md):

    - A set up I personally like based on the analysis above
        ```bash
        > export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek/deepseek-v4-flash
        > export ANTHROPIC_DEFAULT_SONNET_MODEL=anthropic/haiku-4.5
        > export ANTHROPIC_DEFAULT_OPUS_MODEL=anthropic/sonnet-4.6
        ```

- Verify the current configuration:

    ```bash
    > env | grep ANTH | sort
    ANTHROPIC_AUTH_TOKEN=sk-or-v1-...
    ANTHROPIC_BASE_URL=https://openrouter.ai/api
    ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek/deepseek-v4-flash
    ANTHROPIC_DEFAULT_OPUS_MODEL=anthropic/haiku-4.5
    ANTHROPIC_DEFAULT_SONNET_MODEL=anthropic/haiku-4.5
    ```

## Test Claude Code

<!-- TODO(ai_gp): Show commands with /models /doctor -->

### Verifying the Model in Claude Code

- Once Claude Code launches note that the model is the one mapped on Haiku 
  (in the case above, it's `deepseek-v4-flash`) and the access is
  `API Usage Billing` (since we are not going through a Claude plan)
    ```bash
    > claude ...
     ▐▛███▜▌   Claude Code v2.1.158
    ▝▜█████▛▘  deepseek/deepseek-v4-flash · API Usage Billing
      ▘▘ ▝▝    ~/src/xyz
    ```

- Check which model is active with the `/model` command:

    ```
      ❯ /model

      Select model
         Switch between Claude models. Your pick becomes the default for new
         sessions.

             1. Default (recommended)         Use the default model (currently
                anthropic/haiku-4.5) · $5/$25 per Mtok
             2. anthropic/haiku-4.5           Custom Opus model
             3. anthropic/haiku-4.5           Custom Sonnet model
           ❯ 4. deepseek/deepseek-v4-flash ✔  Custom Haiku model
      ```

## Monitoring and Troubleshooting

- Check [OpenRouter Logs](https://openrouter.ai/logs) for request history,
  latency, and error details

- For an alternative workflow using OpenRouter with Simon Willison's LLM CLI,
  see [How to Use OpenRouter with LLM CLI](how_to.Use_OpenRouter.md)

- Refer to the
  [OpenRouter API documentation](https://openrouter.ai/docs/api-reference) for
  the full API specification

- Common issues:
    - **Authentication errors**: Make sure `ANTHROPIC_AUTH_TOKEN` is your
      OpenRouter key, not your Anthropic key
    - **Model not found**: Check the exact model ID on OpenRouter's models page
      (e.g., `deepseek/deepseek-v4-flash`, not `deepseek-v4-flash`)
    - **Rate limiting**: OpenRouter applies rate limits per provider
    - **Key conflicts**: Unset `ANTHROPIC_API_KEY` when using OpenRouter since it
      can conflict with `ANTHROPIC_AUTH_TOKEN`

## Using cc
- This repository also ships a `cc` wrapper script that automates the environment
  variable setup for OpenRouter
