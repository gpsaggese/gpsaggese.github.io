# Building AI Chat Right Into Your Jupyter Notebooks

*How we built a simple Jupyter extension with LangChain in under 100 lines of code*

---

If you've ever worked in a Jupyter notebook for data analysis or research, you know the pattern: write some code, run it, check the output, then switch to another tab to ask ChatGPT or Claude about an error or next steps. It's a constant context switch that breaks your flow.

What if you could just chat with an AI assistant directly in your notebook without leaving Jupyter? And what if building that was actually... simple?

That's exactly what we're going to build: a lightweight chat interface powered by LangChain that lives right inside your Jupyter environment. No complex frameworks, no heavy dependencies just widgets, Python, and a bit of async magic.

---

## Why This Matters

The typical data science workflow involves a lot of back-and-forth:

1. Write code in Jupyter
2. Run into an error or question
3. Copy the error message
4. Open ChatGPT/Claude in a browser
5. Paste and ask
6. Read the response
7. Switch back to Jupyter
8. Try the suggested fix

Each context switch costs time and mental energy. What if step 2 could immediately connect to step 5? That's the promise of embedding a chat interface directly into your notebook environment.

---

## The Vision: Chat as a Jupyter Widget

Jupyter already has a powerful widget system (`ipywidgets`) that lets you build interactive UIs. You've probably seen sliders, buttons, and dropdowns in notebooks. We can use the same system to build a chat interface.

Here's what our final result looks like:

```python
from jupyter_chat import ChatInterface

# Create a chat interface with LangChain
chat = ChatInterface(model="gpt-4")
chat.display()
```

And just like that, you have a chat box in your notebook. Type a message, hit send, and get an AI response all without leaving Jupyter.

The beauty is in the simplicity: no browser extensions, no separate servers, just widgets and Python.

---

## Building Blocks: ipywidgets + LangChain

Our architecture has two main components:

### 1. The UI Layer (ipywidgets)

We need three widget elements:
- An output area to display the conversation
- A text input for typing messages
- A button to send messages

```python
from ipywidgets import Textarea, Button, Output, VBox

class ChatInterface:
    def __init__(self):
        self.output = Output()
        self.input_box = Textarea(placeholder='Type your message...')
        self.send_button = Button(description='Send')
        
        # Wire up the button click
        self.send_button.on_click(self._on_send)
        
    def display(self):
        return VBox([self.output, self.input_box, self.send_button])
```

That's it. When the user clicks "Send", we capture the input and process it.

### 2. The AI Layer (LangChain)

LangChain gives us a clean abstraction for working with LLMs. We can create a simple conversational agent with memory in just a few lines:

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ChatInterface:
    def __init__(self, model="gpt-4"):
        # Initialize LangChain components
        llm = ChatOpenAI(model=model, temperature=0.7)
        memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=llm, memory=memory)
```

The `ConversationBufferMemory` automatically keeps track of the conversation history, so the AI remembers what you talked about earlier in the session.

---

## Handling the Chat Loop

When the user sends a message, we need to:
1. Display the user's message in the output area
2. Call LangChain to get a response
3. Display the AI's response

Here's the simplified version:

```python
def _on_send(self, button):
    user_message = self.input_box.value
    if not user_message.strip():
        return
    
    # Clear input
    self.input_box.value = ''
    
    # Display user message
    with self.output:
        print(f"You: {user_message}")
    
    # Get AI response
    try:
        response = self.chain.run(user_message)
        with self.output:
            print(f"AI: {response}\n")
    except Exception as e:
        with self.output:
            print(f"Error: {e}\n")
```

The `with self.output:` context manager ensures all print statements go to our output widget instead of the main notebook output.

---

## Making It Async (The Right Way)

One problem: LLM API calls can be slow. If we call `chain.run()` synchronously, it blocks the entire notebook kernel. The UI freezes until the response comes back.

The fix? Use async/await:

```python
import asyncio
from langchain.callbacks import AsyncIteratorCallbackHandler

async def _get_response_async(self, message):
    # Run LangChain in async mode
    response = await self.chain.arun(message)
    return response

def _on_send(self, button):
    user_message = self.input_box.value
    # ... display user message ...
    
    # Run async in the event loop
    asyncio.create_task(self._handle_response(user_message))

async def _handle_response(self, message):
    response = await self._get_response_async(message)
    with self.output:
        print(f"AI: {response}\n")
```

Now the notebook stays responsive while waiting for the AI. You can keep working in other cells.

---

## Adding Context: Accessing Notebook Variables

Here's where it gets really powerful: what if the chat could see your notebook's variables?

```python
# In your notebook
df = pd.read_csv('data.csv')
print(df.head())

# In chat:
# User: "What columns are in my dataframe?"
# AI: "Your dataframe has columns: id, name, age, city"
```

We can inject notebook context into the LangChain prompt:

```python
def _build_context(self):
    # Get IPython instance
    from IPython import get_ipython
    ipython = get_ipython()
    
    # Get user variables
    variables = ipython.user_ns
    
    context_items = []
    for name, value in variables.items():
        if not name.startswith('_'):  # Skip private vars
            var_type = type(value).__name__
            context_items.append(f"{name}: {var_type}")
    
    return "\n".join(context_items)

def _on_send(self, button):
    user_message = self.input_box.value
    context = self._build_context()
    
    # Add context to the message
    enhanced_message = f"Context:\n{context}\n\nQuestion: {user_message}"
    # ... send to LangChain ...
```

Now your chat assistant is "aware" of your notebook's state. It's like having a pair programmer who can see your code.

---

## The Complete Implementation

Here's the full working code (seriously, it's less than 100 lines):

```python
import asyncio
from ipywidgets import Textarea, Button, Output, VBox
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ChatInterface:
    def __init__(self, model="gpt-4", temperature=0.7):
        # UI components
        self.output = Output()
        self.input_box = Textarea(
            placeholder='Ask me anything...',
            layout={'width': '100%', 'height': '80px'}
        )
        self.send_button = Button(
            description='Send',
            button_style='primary'
        )
        self.send_button.on_click(self._on_send)
        
        # LangChain setup
        llm = ChatOpenAI(model=model, temperature=temperature)
        memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=llm, memory=memory)
    
    def display(self):
        """Display the chat interface."""
        return VBox([
            self.output,
            self.input_box,
            self.send_button
        ])
    
    def _on_send(self, button):
        """Handle send button click."""
        user_message = self.input_box.value.strip()
        if not user_message:
            return
        
        # Clear input and display user message
        self.input_box.value = ''
        with self.output:
            print(f"You: {user_message}")
        
        # Get response asynchronously
        asyncio.create_task(self._get_and_display_response(user_message))
    
    async def _get_and_display_response(self, message):
        """Get AI response and display it."""
        try:
            response = await self.chain.arun(message)
            with self.output:
                print(f"AI: {response}\n")
        except Exception as e:
            with self.output:
                print(f"Error: {e}\n")

# Usage
chat = ChatInterface()
chat.display()
```

That's it. Copy this into a Jupyter notebook, and you have a working AI chat interface.

---

## Going Further: Tools and Agents

LangChain's real power comes from tools and agents. Want your chat to execute Python code? Search the web? Query a database? Just add tools:

```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import PythonREPLTool

# Define tools
python_tool = PythonREPLTool()

tools = [
    Tool(
        name="Python",
        func=python_tool.run,
        description="Execute Python code"
    )
]

# Create an agent instead of a simple chain
from langchain.agents import AgentType

self.agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory
)
```

Now you can say: "Calculate the mean of [1, 2, 3, 4, 5]" and the agent will execute Python code to answer.

---

## Lessons Learned

Building this taught us a few things:

1. Keep it simple: You don't need a complex framework. Widgets + LangChain = working chat.

2. Async is essential: Blocking calls kill the notebook UX. Always use async for API calls.

3. Context is king: Giving the AI access to notebook variables makes it genuinely useful.

4. Iteration matters: Start with a basic chat, then add features (tools, memory, context) incrementally.

---

## Try It Yourself

The complete tutorial notebook with runnable code is in this repository. Clone it, set your OpenAI API key, and start experimenting:

```bash
git clone <repo-url>
cd tutorials/Jupyter_Extension_Langchain
pip install -r requirements.txt
jupyter lab
```

Open `jupyter_chat_extension.ipynb` and run the cells. Within minutes, you'll have AI chat in your notebook.

---

## What's Next?

Some ideas for extending this:

- Streaming responses: Show the AI's response word-by-word as it generates
- Code execution safety: Sandbox executed code to prevent security issues
- Multi-modal support: Add image/file upload capabilities
- Persistent memory: Save conversation history across sessions
- Custom tools: Add domain-specific tools (e.g., data visualization, SQL queries)

The foundation is simple, but the possibilities are endless.

---

## From Prototype to Production: The JupyterLab Extension

The in-notebook chat we just built is great for prototyping and learning, but what if you want something more production-ready? Something that:

- Doesn't clutter your notebooks with widget cells
- Persists across different notebooks
- Keeps API keys secure on the server
- Can be distributed to your team as a package

That's where a proper JupyterLab extension comes in.

### The Extension Architecture

A JupyterLab extension splits into two parts:

1. Frontend (TypeScript): A sidebar panel that captures notebook context and displays chat UI
2. Backend (Python): A Tornado REST handler that processes messages with LangChain

```
┌──────────────┐   POST /generate    ┌──────────────┐
│  Frontend    │ ───────────────────> │  Backend     │
│  (Browser)   │   {message, context} │  (Server)    │
│              │ <─────────────────── │              │
└──────────────┘   {reply}            └──────────────┘
                                            │
                                            ▼
                                      ┌──────────────┐
                                      │  LangChain   │
                                      │  + OpenAI    │
                                      └──────────────┘
```

### Frontend: Capturing Context Automatically

The TypeScript frontend (`src/panel.ts`) creates a sidebar panel that automatically captures notebook context:

```typescript
export class ContextChatPanel extends Widget {
  private getActiveNotebookContext(): string {
    const current = this._notebooks.currentWidget;
    if (!current?.model?.sharedModel) return '';
    
    const notebook = current.model.sharedModel;
    const activeCell = current.content.activeCell;
    
    // Try active cell first
    if (activeCell?.model.sharedModel.source) {
      return activeCell.model.sharedModel.source;
    }
    
    // Fall back to all cells
    const cells = notebook.cells;
    const sources = [];
    for (let i = 0; i < cells.length; i++) {
      sources.push(cells.get(i).source);
    }
    return sources.join('\n---\n').slice(0, 30000);
  }
}
```

This runs in the browser and captures context without any kernel interaction keeping it fast and secure.

### Backend: Server-Side LLM Calls

The Python backend (`context_chat/handlers.py`) handles the actual LLM interaction:

```python
class GenerateHandler(RequestHandler):
    def post(self):
        data = json.loads(self.request.body)
        message = data.get("message", "")
        context = data.get("context", "")
        
        reply = generate_reply(message, context)
        self.finish({"reply": reply})

def generate_reply(message: str, context: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are an AI assistant embedded in JupyterLab.
    You help users understand and work with their Jupyter notebooks."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message}")
    ]
    
    response = llm.invoke(messages)
    return response.content
```

API keys live on the server they never touch the browser. Much more secure for team deployments.

### Installation and Usage

The extension can be installed and used with minimal setup:

```bash
cd jupyterlab_context_chat
pip install -e .
jlpm install && jlpm build
jupyter labextension develop . --overwrite

export OPENAI_API_KEY="sk-your-key"
jupyter lab
```

Open the Command Palette (Ctrl+Shift+C) and run "Open Context Chat". A sidebar appears with the chat interface no cells to run, no setup in each notebook.

### Key Benefits Over In-Notebook Approach

Why build an extension instead of using widgets?

1. Clean notebooks: No widget cells cluttering your analysis
2. Persistence: Chat available across all notebooks in the session
3. Security: API keys stay on server, never in notebook files
4. Distribution: Package as a Python wheel, install anywhere
5. Performance: Frontend runs independently of kernel
6. Professional UX: Proper sidebar panel instead of cell-based UI

### When to Use Which Approach

In-Notebook (ipywidgets):
- Quick prototypes and experiments
- Learning LangChain concepts
- Single notebook projects
- Demos and tutorials

JupyterLab Extension:
- Production deployments
- Team installations
- Long-term maintainability
- Clean, professional UX
- Secure API key handling

### Migration Strategy

Start with the notebook approach to prototype your prompts and LLM interactions. Once you have something working, port the Python logic to a JupyterLab extension backend. The LangChain code is nearly identical the only difference is where it runs.

---

## Complete Tutorial Files

This repository contains both approaches:

1. `jupyter_chat_extension.ipynb` - In-notebook tutorial with runnable examples
2. `jupyterlab_context_chat/` - Full JupyterLab extension with TypeScript frontend and Python backend

Each approach has its place. Start simple with widgets, scale up to an extension when you need production features.

---

## Getting Started

Clone the repository and choose your path:

```bash
git clone <repo-url>
cd tutorials/Jupyter_Extension_Langchain

# For in-notebook approach:
pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-key"
jupyter lab
# Open jupyter_chat_extension.ipynb

# For JupyterLab extension:
cd jupyterlab_context_chat
pip install -e .
jlpm install && jlpm build
jupyter labextension develop . --overwrite
export OPENAI_API_KEY="sk-your-key"
jupyter lab
# Ctrl+Shift+C → "Open Context Chat"
```

---

## Final Thoughts

Building AI chat into Jupyter doesn't have to be complicated. With ipywidgets and LangChain, you can create a working prototype in under 100 lines. When you need production features, the JupyterLab extension architecture provides a clean separation of concerns and professional UX.

The key insight is this: the AI logic stays the same. Whether running in a notebook cell or a server endpoint, you're still just calling LangChain with a prompt and context. The architecture choice is about where that code lives and how users interact with it.

Start simple. Build something that works. Then scale up when the need arises. That's the path to sustainable AI tools in Jupyter.

---

