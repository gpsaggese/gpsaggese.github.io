# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
pip install bambooai

# %%
import pandas as pd
from bambooai import BambooAI

import plotly.io as pio
from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads the variables
import os
import pandas as pd
from bambooai import BambooAI
# ... rest of your code
pio.renderers.default = 'jupyterlab'

# %%
print(os.getenv("EXECUTION_MODE"))

# %%
df = pd.read_csv('testdata.csv')
bamboo = BambooAI(df=df, planning=True, vector_db=False, search_tool=False)
bamboo.pd_agent_converse()
