# CrewAI API Reference

This document provides a comprehensive reference for all CrewAI API calls used in this project.

---

## Table of Contents

- [Core Imports](#core-imports)
- [Agent API](#agent-api)
- [Task API](#task-api)
- [Crew API](#crew-api)
- [LLM API](#llm-api)
- [Tool API](#tool-api)
- [Process API](#process-api)

---

## Core Imports

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
```

---

## Agent API

### `Agent()` Constructor

Creates an AI agent with specific role, goal, and capabilities.

**Signature:**
```python
Agent(
    role: str,                    # Agent's role/job title
    goal: str,                    # What the agent should accomplish
    backstory: str,               # Agent's background/personality
    verbose: bool = False,        # Print detailed execution logs
    allow_delegation: bool = True, # Allow agent to delegate tasks
    llm: LLM = None,             # LLM instance to use
    tools: List[Tool] = []        # Tools available to the agent
)
```

**Example from `agents.py`:**
```python
from crewai import Agent
from config import get_llm
from tools import get_agent_tools

llm = get_llm()

engineer_agent = Agent(
    role="Data Engineer",
    goal="Process, clean, and prepare data for analysis.",
    backstory="You are an expert data engineer...",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=get_agent_tools(data_path)
)
```

**Key Parameters:**
- `role`: Defines what the agent does (e.g., "Data Engineer", "Data Analyst")
- `goal`: The agent's primary objective
- `backstory`: Context that shapes agent behavior and decision-making
- `verbose`: Set to `True` to see agent's thinking process
- `allow_delegation`: Set to `False` to prevent agents from assigning work to others
- `llm`: Shared LLM instance (all agents can use the same LLM)
- `tools`: List of functions decorated with `@tool` that agent can use

**Agent Methods:**
- `agent.execute_task(task, context)` - Execute a specific task
- `agent.think(prompt)` - Get agent's reasoning (if verbose enabled)

---

## Task API

### `Task()` Constructor

Defines a task that an agent should complete.

**Signature:**
```python
Task(
    description: str,             # What needs to be done
    agent: Agent,                 # Agent assigned to this task
    expected_output: str = "",    # What the output should look like
    context: List[Task] = []      # Tasks that must complete first
)
```

**Example from `tasks.py`:**
```python
from crewai import Task

data_engineering_task = Task(
    description="""
    Quickly examine the dataset located at {data_path}.
    Your tasks:
    1. Get a brief summary of the dataset structure
    2. Note the key columns available
    3. Verify the data is ready for analysis
    """,
    agent=engineer_agent,
    expected_output="A brief confirmation that the dataset is loaded and ready for analysis."
)
```

**Key Parameters:**
- `description`: Detailed instructions for what the agent should do
- `agent`: The agent responsible for completing this task
- `expected_output`: Description of what the result should contain
- `context`: List of tasks that must complete before this task runs (for dependencies)

**Task Dependencies:**
```python
# Task 2 depends on Task 1
task2 = Task(
    description="Analyze the data",
    agent=analyst_agent,
    context=[task1]  # Task 1 must complete first
)

# Task 3 depends on Task 2
task3 = Task(
    description="Create story from analysis",
    agent=storyteller_agent,
    context=[task2]  # Task 2 must complete first
)
```

**Parallel Execution:**
```python
# Tasks without context run in parallel
task1 = Task(description="...", agent=agent1)  # No context
task2 = Task(description="...", agent=agent2)  # No context
# Both run simultaneously
```

---

## Crew API

### `Crew()` Constructor

Orchestrates agents and tasks into a workflow.

**Signature:**
```python
Crew(
    agents: List[Agent],          # Agents in the crew
    tasks: List[Task],            # Tasks to execute
    process: Process = Process.sequential,  # Execution process
    verbose: bool = False,        # Print execution details
    memory: bool = False,         # Enable memory between runs
    max_iter: int = 15,          # Max iterations per task
    max_execution_time: int = None  # Max execution time in seconds
)
```

**Example from `crew.py`:**
```python
from crewai import Crew, Process

crew = Crew(
    agents=[engineer_agent, analyst_agent, storyteller_agent],
    tasks=[data_engineering_task, custom_analysis_task, storyteller_task],
    process=Process.sequential,
    verbose=True
)
```

**Key Parameters:**
- `agents`: List of all agents that will participate
- `tasks`: List of tasks to execute (order matters for dependencies)
- `process`: Execution strategy (see Process API below)
- `verbose`: Print detailed execution logs
- `memory`: Enable agents to remember previous conversations
- `max_iter`: Maximum number of iterations per task (prevents infinite loops)

**Crew Methods:**

#### `crew.kickoff()`
Starts the crew execution and returns results.

**Returns:**
```python
class CrewOutput:
    tasks_output: List[str]  # Output from each task
    raw: str                 # Raw output string
```

**Example:**
```python
crew = create_flow_crew(user_query, csv_path)
result = crew.kickoff()

# Access individual task outputs
engineer_output = result.tasks_output[0]
analyst_output = result.tasks_output[1]
storyteller_output = result.tasks_output[2]

# Or get full output
full_output = str(result)
```

**Execution Flow:**
1. CrewAI analyzes task dependencies
2. Tasks without dependencies run in parallel
3. Dependent tasks wait for their context tasks
4. Results are aggregated and returned

---

## LLM API

### `LLM()` Constructor

Configures the Language Model for agents.

**Signature:**
```python
LLM(
    model: str,                  # Model identifier
    api_key: str = None,         # API key for the provider
    base_url: str = None,        # Custom API endpoint
    temperature: float = 0.7,    # Creativity (0.0-1.0)
    max_tokens: int = None       # Max tokens per response
)
```

**Example from `config.py`:**
```python
from crewai import LLM

# Hugging Face
llm = LLM(
    model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=HF_API_KEY
)

# Ollama (local)
llm = LLM(
    model="mistral",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# OpenRouter
llm = LLM(
    model="openrouter/google/gemma-2-2b-it:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.3
)
```

**Supported Providers:**
- `huggingface/{model}` - Hugging Face Inference API
- `openrouter/{model}` - OpenRouter API
- `ollama` - Local Ollama (via base_url)
- `openai` - OpenAI API (default)

**LLM Configuration:**
- `model`: Provider-specific model identifier
- `api_key`: Authentication key (required for most providers)
- `base_url`: Custom endpoint (for Ollama, custom APIs)
- `temperature`: Controls randomness (lower = more focused, higher = more creative)

---

## Tool API

### `@tool` Decorator

Makes a function available to agents as a tool.

**Signature:**
```python
@tool("tool_name")
def tool_function(param1: type, param2: type = default) -> str:
    """
    Tool description that agent sees.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    
    Returns:
        Description of return value
    """
    # Implementation
    return result_string
```

**Example from `tools.py`:**
```python
from crewai.tools import tool

@tool("read_nba_data")
def read_nba_data(limit: int = 10) -> str:
    """
    Read a sample of the NBA data file to understand its structure.
    
    Args:
        limit: Number of sample rows to return (default: 10, max: 50)
    """
    df = pd.read_csv(data_path)
    sample = df.head(limit)
    return f"Dataset: {len(df):,} total records...\n\n{sample.to_string()}"
```

**Tool Requirements:**
1. Must be decorated with `@tool("name")`
2. Must have a docstring (agents read this)
3. Must return a string (agents process text)
4. Type hints help agents understand parameters
5. Docstring should explain what the tool does

**Tool Usage by Agents:**
- Agents automatically see available tools
- Agents decide when to use tools based on task
- Tool results are passed back to agent's LLM
- Agent processes tool output and continues reasoning

---

## Process API

### `Process` Enum

Defines how tasks are executed.

**Options:**
```python
from crewai import Process

Process.sequential    # Execute tasks one by one (respects dependencies)
Process.hierarchical  # Manager-agent delegates to worker-agents
```

**Example:**
```python
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential  # Default
)
```

**Process.sequential:**
- Tasks execute in order
- Dependencies are respected
- Independent tasks can run in parallel
- Most common for data analysis workflows

**Process.hierarchical:**
- Manager agent coordinates
- Worker agents execute subtasks
- Useful for complex delegation scenarios

---

## Complete API Usage Example

### Creating a Full Crew

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. Configure LLM
llm = LLM(
    model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=os.getenv("HF_API_KEY")
)

# 2. Create Tools
@tool("get_data_summary")
def get_data_summary() -> str:
    """Get summary of the dataset."""
    return "Dataset has 5000 rows, 15 columns..."

tools = [get_data_summary]

# 3. Create Agents
engineer = Agent(
    role="Data Engineer",
    goal="Process and validate data",
    backstory="Expert data engineer...",
    llm=llm,
    tools=tools,
    verbose=True
)

analyst = Agent(
    role="Data Analyst",
    goal="Extract insights from data",
    backstory="Seasoned analyst...",
    llm=llm,
    tools=tools,
    verbose=True
)

# 4. Create Tasks
task1 = Task(
    description="Validate the dataset structure",
    agent=engineer,
    expected_output="Confirmation that data is ready"
)

task2 = Task(
    description="Analyze the data and find top performers",
    agent=analyst,
    context=[task1],  # Depends on task1
    expected_output="Analysis report with insights"
)

# 5. Create Crew
crew = Crew(
    agents=[engineer, analyst],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

# 6. Execute
result = crew.kickoff()
print(result.tasks_output[0])  # Engineer output
print(result.tasks_output[1])  # Analyst output
```

---

## API Best Practices

### 1. Shared LLM Instance
```python
# Good: Create once, reuse
llm = get_llm()
agent1 = Agent(..., llm=llm)
agent2 = Agent(..., llm=llm)

# Bad: Creating multiple instances
agent1 = Agent(..., llm=get_llm())
agent2 = Agent(..., llm=get_llm())
```

### 2. Tool Documentation
```python
# Good: Clear docstring
@tool("analyze_data")
def analyze_data(query: str) -> str:
    """
    Analyze data based on query.
    
    Args:
        query: What to analyze (e.g., "top 5 players")
    
    Returns:
        Analysis results as formatted string
    """
    ...

# Bad: No documentation
@tool("analyze")
def analyze(q): return "..."
```

### 3. Task Dependencies
```python
# Good: Clear dependencies
task2 = Task(..., context=[task1])
task3 = Task(..., context=[task2])

# Bad: Circular or missing dependencies
task2 = Task(..., context=[task3])  # Circular!
task3 = Task(...)  # Missing dependency
```

### 4. Error Handling
```python
# Good: Handle errors
try:
    result = crew.kickoff()
except Exception as e:
    print(f"Error: {e}")
    # Handle gracefully

# Bad: No error handling
result = crew.kickoff()  # May crash
```

---

## API Reference Summary

| Component | Class/Function | Key Methods | Purpose |
|-----------|---------------|-------------|---------|
| **Agent** | `Agent()` | `execute_task()` | AI agent with role and tools |
| **Task** | `Task()` | - | Work item for an agent |
| **Crew** | `Crew()` | `kickoff()` | Orchestrates agents and tasks |
| **LLM** | `LLM()` | - | Language model configuration |
| **Tool** | `@tool()` | - | Function decorator for agent tools |
| **Process** | `Process` enum | - | Execution strategy |

---

## Common Patterns

### Pattern 1: Sequential Analysis
```python
# Task 1 → Task 2 → Task 3
task1 = Task(..., agent=agent1)
task2 = Task(..., agent=agent2, context=[task1])
task3 = Task(..., agent=agent3, context=[task2])
```

### Pattern 2: Parallel + Sequential
```python
# Task 1 and Task 2 run in parallel
# Task 3 waits for both
task1 = Task(..., agent=agent1)
task2 = Task(..., agent=agent2)
task3 = Task(..., agent=agent3, context=[task1, task2])
```

### Pattern 3: Independent Parallel
```python
# All tasks run simultaneously
task1 = Task(..., agent=agent1)
task2 = Task(..., agent=agent2)
task3 = Task(..., agent=agent3)
# No context = parallel execution
```

---

**For more details, see:**
- CrewAI Official Docs: https://docs.crewai.com
- Example Flow: `crewai.example.py`
- Execution Flow: `EXECUTION_FLOW.md`
