For the notebooks in

msml610/tutorials/L03_knowledge_representation

Add a cell after 

!pip install -q python-sat sympy z3-solver

import pysat
print("pysat version: ", pysat.__version__)
import sympy==
print("sympy version: ", sympy.__version__)
import z3
print("z3 version: ", z3.get_version_string())

