# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut
import utils_Lesson05_1_Learning_Theory_VC_Dimension as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Cell 1: Dichotomy Explorer - 2D Perceptron with 3 Points
#
# - Explore how a 2D perceptron (a separating line) can classify 3 points in different ways:
#   - Visualize 3 labeled points (A, B, C) in a 2D plane
#   - Adjust the separating line by changing its angle and offset
#   - Observe how the classification of each point changes (blue for +1, red for -1)
#   - Discover all possible dichotomies (different ways to partition the points)
#
# **Parameters**:
# - `Point Config`: Choice of point configurations
#   - `collinear1`, `collinear2`: Points arranged in a line
#   - `triangle1`, `triangle2`, `triangle3`: Points arranged in a triangle
# - `angle`: Angle of the line normal in degrees (0 to 360)
# - `offset`: Distance of the separating line from the origin (-1.5 to 1.5)
#
# **Key observation**:
# - For 3 points, there are $2^3 = 8$ possible dichotomies (ways to assign +1/-1 labels)
# - Try different angle and offset combinations to discover all possible classifications
# - Some dichotomies may be easier to find than others depending on the point configuration
# - This visualization helps understand the concept of VC dimension - the maximum number of points that can be shattered (all dichotomies realized) by a hypothesis class

# %%
# Explore how a 2D perceptron can classify 3 points in different ways.
# Adjust the angle and offset of the separating line to discover all possible dichotomies.
utils.cell1_dichotomy_explorer_3points()
# Try different angles and offsets to discover all 8 possible classifications of 3 points.

# %% [markdown]
# ## Cell 2: Dichotomy Explorer - 2D Perceptron with 3 Points (Target Assignment)
#
# - Discover that 3 points can be classified in $2^3 = 8$ different ways:
#   - Select a target classification (one of 8 possible assignments)
#   - Visualize the 3 points colored according to the target (blue for +1, red for -1)
#   - Adjust the separating line to match the target classification
#   - Use "Find Solution" button to automatically discover a valid configuration
#
# **Parameters**:
# - `Point Config`: Choice of point configurations
#   - `collinear1`, `collinear2`: Points arranged in a line
#   - `triangle1`, `triangle2`, `triangle3`: Points arranged in a triangle
# - `Target`: Select one of 8 possible target classifications (Assignment 0-7)
# - `angle`: Angle of the line normal in degrees (0 to 360)
# - `offset`: Distance of the separating line from the origin (-1.5 to 1.5)
# - `Find Solution`: Button to automatically find a line that achieves the target
#
# **Key observation**:
# - All 8 dichotomies can be achieved for 3 points in general position
# - The "Find Solution" button demonstrates that each target is realizable
# - When points are collinear, some dichotomies may still be achievable but require careful positioning
# - This shows that 3 points can be "shattered" by a 2D perceptron in most configurations

# %%
# Discover that 3 points can be classified in 2^3 = 8 different ways.
# Select a target classification and adjust the line to match it.
utils.cell2_dichotomy_explorer_3points_target()
# Use 'Find Solution' to automatically discover a line that achieves the target.

# %% [markdown]
# ## Cell 3: Dichotomy Explorer - 2D Perceptron with 4 Points
#
# - Explore the limitations of linear separators with 4 points:
#   - Visualize 4 labeled points (A, B, C, D) in different configurations
#   - Adjust the separating line to discover different classifications
#   - Discover that not all $2^4 = 16$ classifications are achievable
#
# **Parameters**:
# - `Point Config`: Choice of 4-point configurations
#   - `square`: Points arranged in a square (reveals XOR limitation)
#   - `circle`: Points arranged in a circle
#   - `line`: Points arranged in a line
#   - `diamond`: Points arranged in a diamond shape
# - `angle`: Angle of the line normal in degrees (0 to 360)
# - `offset`: Distance of the separating line from the origin (-1.5 to 1.5)
#
# **Key observation**:
# - With 4 points, only 14 out of 16 dichotomies are achievable
# - The XOR pattern (opposite corners same color in square) is impossible
# - This introduces the concept of **break point**: $k = 4$ for 2D perceptron
# - Since not all dichotomies are achievable, the growth function $m_H(4) = 14 < 2^4 = 16$
# - This limitation is fundamental to linear separators and leads to the VC dimension concept

# %%
# Explore how 4 points reveal the break point for 2D perceptrons.
# Try different angles and offsets to find unique dichotomies.
utils.cell3_dichotomy_explorer_4points()
# Notice that you can find at most 14 out of 16 possible classifications.

# %%
