# Conversational Diagram Designer (CDD)

## Overview
Conversational Diagram Designer (CDD) is a browser-based diagramming tool that
allows users to create and refine diagrams using natural language.

It combines:

- A diagram code editor
- A live renderer
- A chat interface powered by LLMs
- A vision feedback loop (rendered image sent back to the LLM)

CDD supports:

- Graphviz (DOT)
- Mermaid
- C4 (PlantUML or Structurizr DSL)

The system can:

- Run fully locally in the browser
- Be deployed to AWS and exposed as a web application

It's like
[https://dreampuf.github.io/GraphvizOnline/](https://dreampuf.github.io/GraphvizOnline/)?
but with a chat to have LLM (and other models) help create the graph

## Goals

### Primary Goals
- Conversational diagram creation and editing
- Real-time rendering
- LLM-driven diagram modification
- Vision-based diagram validation
- Local and cloud deployment options

### Non-Goals (V1)
- Multi-user collaboration
- Enterprise authentication
- Persistence

## Core User Flow
1. User types:

   "Create a microservices architecture with API Gateway, 3 services, and a
   database."

2. LLM generates diagram source code (Mermaid/DOT/C4).

3. Renderer converts source → SVG/PNG.

4. Rendered image is sent back to the LLM for visual validation.

5. User iterates:

   "Move database to the bottom and show replication."

6. LLM updates the full diagram source.

7. System re-renders.

8. Loop continues.

## High-Level Architecture
```
Browser UI
├── Chat Panel
├── Code Editor
├── Diagram Renderer
└── Vision Feedback Engine

Optional Backend (AWS)
├── LLM Proxy
├── Model Router
└── Storage (optional)
```

## Frontend Architecture

### Recommended Stack
- React / Next.js
- Monaco Editor
- Mermaid.js
- Viz.js (Graphviz WASM)
- Optional PlantUML rendering
- OpenAI/Anthropic/local LLM APIs

## Diagram Engine

### Supported Formats
| Format        | Renderer       | Local Support |
| :------------ | :------------- | :------------ |
| Mermaid       | Mermaid.js     | Yes           |
| Graphviz DOT  | Viz.js (WASM)  | Yes           |
| C4 (PlantUML) | Server or WASM | Partial       |

## LLM Interaction Model

### System Prompt (Example)
You are a diagram engineer. You output only valid diagram code. When modifying,
output the FULL updated diagram. Never include explanations unless explicitly
requested.

### Operation Modes
- Create
- Modify
- Debug
- Refactor
- Explain

## Vision Feedback Loop

### Purpose
LLMs struggle with spatial reasoning unless they see the rendered output.

Vision feedback enables:

- Layout validation
- Detection of overlapping nodes
- Missing connections
- Visual hierarchy issues
- Logical inconsistencies

### Flow
1. Render diagram → SVG/PNG.
2. Convert to base64 image.
3. Send image to multimodal LLM.
4. LLM evaluates layout and returns corrected full diagram code.
5. System re-renders.

Limit auto-correction to 3 iterations to avoid infinite loops.

### Example of a Conversation
An example of a conversation is

1) Create a graphviz graph based on this content
- **System**: object you want to estimate/track
- **Filter**: algorithm to estimate the state of the system
- **State of the system** x: current values you are interested in
  - Part of the state might be hidden (i.e., only partially observable)
- **Measurement** z: the measured value of the system
  - Observable, but it can be inaccurate
- **State estimate** x_est: filter estimate of the state
- **System model**: mathematical model of the system
  - Typically there is error in the specification of the model
- **System propagation**: predict step using the system model to form a new
  state estimate x_pred
  - Because the system model and the measurements are imperfect, the estimate
    is imperfect
- **Measurement update**: update step
There should be a System and a Filter on two rows stacked, where Filter estimates the state of System
```

2) Write the variables that correspond to a circle on top of the edges

3) No need to have the circle, just write the names of the variables on the edges

4) Put Predict step, Update step, system model, Filter inside a subgraph to keep it together
