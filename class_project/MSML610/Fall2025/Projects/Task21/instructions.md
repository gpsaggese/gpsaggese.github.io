Project 3: Multi-Agent Cooperation (Difficulty: 3)
Project Objective:
Design a multi-agent reinforcement learning system where agents must collaborate to achieve a common goal, optimizing for collective performance.

Dataset Suggestions:

Use the Multi-Agent Particle Environment (MPE), available on GitHub, which provides a variety of scenarios for multi-agent tasks.
Tasks:

Set up the MPE environment and integrate it with TorchRL.
Implement a centralized training approach using A3C for multiple agents working together.
Train the agents to complete tasks such as gathering resources or reaching a target location.
Evaluate the cooperative performance by measuring success rates and communication efficiency.
Bonus Ideas (Optional):

Investigate different communication strategies between agents.
Compare the effectiveness of centralized vs. decentralized training methods.



https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/instructions/README.md#working-on-the-project

https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/instructions/README.md#submission



Your submission must include the following files:

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

https://github.com/gpsaggese-org/umd_classes/tree/master/class_project/instructions/tutorial_template/tutorial_github_data605_style