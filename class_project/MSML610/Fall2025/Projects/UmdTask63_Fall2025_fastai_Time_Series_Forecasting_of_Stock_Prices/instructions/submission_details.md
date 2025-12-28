# Utils Module:

This file is meant to contain helper functions, reusable logic, and API wrappers.
Keep the notebooks focused on documentation and outputs. Place any logic or workflow functions inside this module.
Scripts/Notebooks:

You will work on one API file and one Example (Your project) file.
We encourage you to use Python files (Utils module) and call the code from notebooks.
Markdowns:

One markdown file linked to each python script, i.e, API and example.
For more guidance on this structure and the rationale behind it, see How to write the Tutorial

# In general

For API: you are expected to describe the API, its architecture, etc.
For Example: You are expected to use the project tool according to the specifications mentioned in the project description
Submission

# Your submission must include the following files:

Important: "API" here refers to the tool's internal interface—not an external data‑provider API. Please keep the focus on the tool itself.

XYZ.API.md:

Document the native programming interface (classes, functions, configuration objects) of your chosen tool or library.
Describe the lightweight wrapper layer you have written on top of this native API.
XYZ.API.ipynb:

A Jupyter notebook demonstrating usage of the native API and your wrapper layer, with clean, minimal cells
XYZ.example.md:

A markdown file presenting a complete example of an application that uses your API layer
XYZ.example.ipynb:

A Jupyter notebook corresponding to the example above, demonstrating end-to-end functionality
XYZ_utils.py:

A Python module containing reusable utility functions and wrappers around the API
The notebooks should invoke logic from this file instead of embedding complex code inline
Difference between {project}.API.* and {project}.example.*

{project}.API.*: stable contract‑only layer. Holds dataclasses, enums, and abstract service interfaces so anyone can integrate without pulling in your runtime code.
```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class User:
    id: int
    email: str

class AuthService(Protocol):
    """Authenticate users without revealing storage details."""
    def register(self, user: User, password: str) -> None: ...
    def login(self, email: str, password: str) -> str: ...  # returns JWT
{project}.example.*: runnable reference implementation that satisfies the API with real storage, I/O, and third‑party calls.

import sqlite3
import bcrypt
import jwt
from project.API.auth import User, AuthService

class SqliteAuthService(AuthService):
    _DB = "users.db"

    def register(self, user: User, password: str) -> None:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        with sqlite3.connect(self._DB) as conn:
            conn.execute(
                "INSERT INTO users(id, email, password) VALUES (?,?,?)",
                (user.id, user.email, hashed),
            )

    def login(self, email: str, password: str) -> str:
        with sqlite3.connect(self._DB) as conn:
            row = conn.execute("SELECT password FROM users WHERE email=?", (email,)).fetchone()
        if not row or not bcrypt.checkpw(password.encode(), row[0]):
            raise PermissionError("invalid credentials")
        return jwt.encode({"sub": email}, "supersecret", algorithm="HS256")
```

# Folder Structure

```
COURSE_CODE/
└── Term20xx/
    └── projects/
        └── TutorTaskXX_Name_of_issue/
            ├── utils_data_io.py
            ├── utils_post_processing.py
            ├── API.ipynb
            ├── API.md
            ├── example.ipynb
            ├── example.md
            ├── Dockerfile
            └── README.md
```

# Submission Guidelines

Each markdown file should explain the intent and design decisions:

Avoid copy-pasting code cells or raw outputs from the notebooks
Instead, use the markdown to communicate the reasoning behind your choices
Each notebook should:

Be self-contained and executable from top to bottom via "Restart and Run All"
Use functions from XYZ_utils.py to keep the cells concise and maintainable
Demonstrate functionality clearly and logically with clean, commented outputs
Docker setup:

Include clear instructions on how to build and run your Docker container
Mention expected terminal outputs when running scripts (e.g., starting Jupyter, mounting volumes, etc.) E.g.,

### To Build the Image

``` <- triple backticks here bash docker_build.sh ```

### To Run the Container

``` bash docker_bash.sh ```

# Visual documentation:

Include diagrams and flowcharts when relevant (e.g., using mermaid) E.g.,

Provide schema descriptions if your project uses a database or structured data E.g.,

Projects that do not run end-to-end or lack proper documentation will be considered incomplete

In case of issues, they will be flagged through GitHub issues, and you will be expected to resolve them in a timely manner