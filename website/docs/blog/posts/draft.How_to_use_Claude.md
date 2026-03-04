// https://platform.claude.com/docs/en/home

# Developer Guide

## Models & Pricing

## Build with Claude

### Prompting best practices

- Use clear and explicit instructions
  - Do not rely on the model to infer from vague prompts
  - Think of C as a brilliant but new employee
  - Be specific about output and constraints
  - Provide instructions as sequential steps

- Use examples
  - Aka few-shot prompting
  - Create examples that are relevant and diverse
  - Wrap examples in `<examples>` tags
  - You can ask C to evaluate examples and provide additional ones

- Use XML tags
  - XML tags help C not get confused with instructions, context, input
  - E.g.,
    ```
    <documents>
    <document index=1>
    ...
    ```

- Give C a role
  - Use the system prompt to focus C's behavior and tone
    ```
    You are a helpful coding assistant specializing in Python
    ```

- When there are large docs
  - Put longform data at the top
  - Put the query at the end
  - Structure document with XML tags
  - Ask C to quote relevant parts of the documents first before carrying out its
    tasks

- Control the format of responses
  - Tell C what to do and what not to do
  - Use XML format indicators
  - Match the prompt style to the desired output

### Tool usage

### Optimize parallel tool calling

- Run multiple speculative searches during research
- Read several files at once to build context faster

## Model Capabilities

### Extended thinking

### Adaptive thinking

### Effort

### Fast mode

### Structure outputs

### Citations

### Streaming messages

### Batch processing

### PDF support

### Search results

### Multilingual support

### Embeddings

### Vision

## Tools

## Tool Infrastructure

## Context Management

## Files & Assets

## Agent Skills

## Agent SDK

## Prompt Engineering

## Strengthen Guardrails

## Admin and Monitoring

# API reference

# MCP

# Resources

# Release Notes
