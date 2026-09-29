# Conversational Diagram Designer (CDD)

## Status
- **Status:**: in_progress
- **Complete Specs:**: 0-100%
- **Assignee:**: ...

## Core Idea

- Conversational Diagram Designer (CDD) is a browser-based diagramming tool
  that lets users create and refine diagrams using natural language
- It combines:
  - A diagram code editor
  - A live renderer
  - A chat interface powered by LLMs
  - A vision feedback loop (the rendered image is sent back to the LLM)
- CDD supports Graphviz (DOT), Mermaid, and C4 (PlantUML or Structurizr DSL)
- The system can run fully locally in the browser, or be deployed to AWS and
  exposed as a web application
- It is similar to
  [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline/), but with a
  chat interface where an LLM (or another model) helps build the graph
- Goals for V1:
  - Conversational diagram creation and editing
  - Real-time rendering
  - LLM-driven diagram modification
  - Vision-based diagram validation
  - Local and cloud deployment options
- Non-goals for V1: multi-user collaboration, enterprise authentication,
  persistence

## Formalization

### High-Level Architecture

```text
Browser UI
- Chat Panel
- Code Editor
- Diagram Renderer
- Vision Feedback Engine

Optional Backend (AWS)
- LLM Proxy
- Model Router
- Storage (optional)
```

### Core User Flow (Turn Sequence)

1. User describes the diagram:
   ```text
   prompt> Create a microservices architecture with API Gateway, 3 services,
   prompt> and a database.
   ```
2. LLM generates diagram source code (Mermaid, DOT, or C4)
3. Renderer converts the source to SVG or PNG
4. Rendered image is sent back to the LLM for visual validation
5. User iterates:
   ```text
   prompt> Move database to the bottom and show replication.
   ```
6. LLM updates the full diagram source
7. System re-renders the diagram
8. Loop continues

### LLM Interaction Model

- System prompt (example):
  ```text
  prompt> You are a diagram engineer. You output only valid diagram code.
  prompt> When modifying, output the FULL updated diagram. Never include
  prompt> explanations unless explicitly requested.
  ```
- Operation modes: create, modify, debug, refactor, explain

### Vision Feedback Loop

- LLMs struggle with spatial reasoning unless they see the rendered output
- Vision feedback enables:
  - Layout validation
  - Detection of overlapping nodes
  - Missing connections
  - Visual hierarchy issues
  - Logical inconsistencies
- Flow:
  1. Render diagram to SVG or PNG
  2. Convert to a base64 image
  3. Send the image to a multimodal LLM
  4. LLM evaluates the layout and returns corrected full diagram code
  5. System re-renders
- Limit auto-correction to 3 iterations to avoid infinite loops

## Key Examples

- **Kalman filter diagram, built conversationally**: a full conversation
  showing the turn-by-turn chat workflow
  1. User pastes the diagram content and the desired layout:
     ```text
     Create a graphviz graph based on this content
     - **System**: object you want to estimate/track
     - **Filter**: algorithm to estimate the state of the system
     - **State of the system** x: current values you are interested in
       - Part of the state might be hidden (i.e., only partially observable)
     - **Measurement** z: the measured value of the system
       - Observable, but it can be inaccurate
     - **State estimate** x_est: filter estimate of the state
     - **System model**: mathematical model of the system
       - Typically there is error in the specification of the model
     - **System propagation**: predict step using the system model to form a
       new state estimate x_pred
       - Because the system model and the measurements are imperfect, the
         estimate is imperfect
     - **Measurement update**: update step
     There should be a System and a Filter on two rows stacked, where Filter estimates the state of System
     ```
  2. User:
     ```text
     prompt> Write the variables that correspond to a circle on top of the
     prompt> edges
     ```
  3. User:
     ```text
     prompt> No need to have the circle, just write the names of the
     prompt> variables on the edges
     ```
  4. User:
     ```text
     prompt> Put Predict step, Update step, system model, Filter inside a
     prompt> subgraph to keep it together
     ```

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: diagram editor and renderer
  - Build the core stack: React/Next.js, Monaco Editor for the code editor
  - Support these diagram formats:

    | Format        | Renderer       | Local Support |
    | :------------ | :------------- | :------------ |
    | Mermaid       | Mermaid.js     | Yes           |
    | Graphviz DOT  | Viz.js (WASM)  | Yes           |
    | C4 (PlantUML) | Server or WASM | Partial       |

  - This is the result: a working code editor and live renderer for Mermaid,
    Graphviz DOT, and C4 diagrams

- Milestone 2: conversational diagram creation and editing
  - Add a chat panel that calls an LLM (OpenAI, Anthropic, or a local model)
  - LLM outputs only valid diagram code, and outputs the FULL updated diagram
    on every edit
  - Support the operation modes: create, modify, debug, refactor, explain
  - This is the result: users can create and edit diagrams by describing
    changes in natural language

- Milestone 3: vision feedback loop
  - Render the diagram, convert it to a base64 image, and send it to a
    multimodal LLM for layout validation
  - Limit auto-correction to 3 iterations to avoid infinite loops
  - This is the result: the LLM catches overlapping nodes, missing
    connections, and visual hierarchy issues before showing the diagram to
    the user

- Milestone 4: optional cloud deployment
  - Deploy an optional AWS backend with an LLM proxy, a model router, and
    optional storage
  - This is the result: CDD can run fully locally in the browser, or be
    deployed to AWS as a hosted web application

## References
- Author(s), _Title_. (Year)
