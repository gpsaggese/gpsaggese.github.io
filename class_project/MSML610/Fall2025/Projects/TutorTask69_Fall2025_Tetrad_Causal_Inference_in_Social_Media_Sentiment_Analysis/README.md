<!-- toc -->
- [Project Files](#project-files)
- [Setup and Dependencies](#setup-and-dependencies)
    * [Running the Project](#running-the-project)
    * [Requirements](#requirements)
- [Sources](#sources)
<!-- tocstop -->

## Project Files
`README.md`: This file, explains how to run this project

`tetrad.API.md`: A description of Tetrad's basic functionality

`tetrad.API.ipynb`: Code for and explanations of Tetrad functions

`tetrad.example.md`: A description of the Tetrad example code

`tetrad.example.ipynb`: An extended example of using Tetrad to analyze causal relationships in a dataset

## Setup and Dependencies
### Running the Project
Navigate to the `dockerfiles` folder contained in this folder and build the docker image.
```bash
> ./docker_build.sh
```

Launch the jupyter notebook.
```bash
> ./docker_jupyter.sh
```

Open the jupyter notebook by navigating to `http://localhost:8888` in your browser.

### Requirements
The following requirements will be satisified with this project's Docker container or inline in the jupyter notebooks:

- Java JDK 21+ 
- Python 3.12+
- py-tetrad
- JPype
- PyTorch

## Sources
- [Tetrad](https://www.cmu.edu/dietrich/philosophy/tetrad/index.html)
- [Official Tetrad Manual](https://htmlpreview.github.io/?https:///github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/resources/docs/manual/index.html)
- [py-tetrad GitHub](https://github.com/cmu-phil/py-tetrad)
- [Tetrad Javadocs](https://www.phil.cmu.edu/tetrad-javadocs/7.6.9/)