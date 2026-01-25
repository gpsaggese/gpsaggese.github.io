# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %% [markdown]
# ### Install packages

# %%
if False:
    pass
    # !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
    # !jupyter labextension enable

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet imageio[ffmpeg])"

# %% [markdown]
# ### Import modules

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ipywidgets import interact, FloatSlider, IntSlider, widgets
from IPython.display import display, Markdown

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %%
import helpers.hmatplotlib as hmatplo
import msml610_utils as ut
import utils_Lesson94_Information_Theory as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


# %% [markdown]
# ## Interactive Visualization: Joint Entropy
#
# Adjust the dependence slider to see how correlation between X and Y affects joint entropy.
# The scatter plot shows sampled realizations from the joint distribution.

# %%
## Create Video from Interactive Widget

import numpy as np

# Parameters for video generation.
n_steps = 11  # This gives us 11 frames from 0.0 to 1.0 (inclusive)
n_samples_fixed = 100  # Fixed sample size
dependence_values = np.linspace(0.0, 1.0, n_steps)

# Prepare values list for generate_animation.
values = []
for val in dependence_values:
    values.append({"dependence": val, "n_samples": n_samples_fixed})

# Directory to save frames.
dst_dir = "./video_frames"

# Generate animation frames with fixed dimensions.
ut.generate_animation(
    utils.plot_joint_entropy_interactive,
    values,
    dst_dir,
    figsize=(20, 5),
    dpi=150
)


# %% [markdown]
# ## Video Generation from Interactive Widget
#
# This section demonstrates how to create a video animation from the interactive widget.
#
# The process:
# 1. Generate frames by calling `plot_joint_entropy_interactive` with dependence values from 0.0 to 1.0 in 11 steps (giving us 10 intervals)
# 2. Save each frame as a PNG image
# 3. Combine frames into an MP4 video using imageio or matplotlib.animation
#
# The resulting video will show how joint entropy changes as the dependence between X and Y varies from complete independence (0.0) to perfect dependence (1.0).

# %%
# Animation generation function has been implemented in msml610_utils.py

# %%
# ## Combine Frames into Video

# try:
#     import imageio
#     print("Using imageio for video creation")
    
#     # Get all frame files sorted by name.
#     frame_files = sorted(frames_dir.glob("frame_*.png"))
#     print(f"Found {len(frame_files)} frames")
    
#     # Read frames.
#     frames = []
#     for frame_file in frame_files:
#         frames.append(imageio.imread(frame_file))
    
#     # Create video with fps (frames per second).
#     # Lower fps = slower video, higher fps = faster video.
#     fps = 2  # 2 frames per second means each frame shows for 0.5 seconds.
#     output_video = "joint_entropy_animation.mp4"
    
#     imageio.mimsave(output_video, frames, fps=fps, codec='libx264')
#     print(f"\nVideo created successfully: {output_video}")
#     print(f"Video duration: {len(frames)/fps:.1f} seconds")
#     print(f"Frame rate: {fps} fps")
    
# except ImportError:
#     print("imageio not installed. Trying alternative method with matplotlib.animation...")
    
#     try:
#         from matplotlib import animation
#         from IPython.display import HTML
        
#         # Get all frame files sorted by name.
#         frame_files = sorted(frames_dir.glob("frame_*.png"))
#         print(f"Found {len(frame_files)} frames")
        
#         # Read first frame to get dimensions.
#         first_frame = plt.imread(frame_files[0])
        
#         # Create figure and axis.
#         fig, ax = plt.subplots(figsize=(20, 5))
#         ax.axis('off')
        
#         # Display first frame.
#         im = ax.imshow(first_frame)
        
#         def update_frame(frame_idx):
#             """Update function for animation."""
#             img = plt.imread(frame_files[frame_idx])
#             im.set_array(img)
#             return [im]
        
#         # Create animation.
#         anim = animation.FuncAnimation(
#             fig,
#             update_frame,
#             frames=len(frame_files),
#             interval=500,  # 500ms = 0.5 seconds per frame
#             blit=True,
#             repeat=True
#         )
        
#         # Save as mp4.
#         output_video = "joint_entropy_animation.mp4"
#         Writer = animation.writers['ffmpeg']
#         writer = Writer(fps=2, metadata=dict(artist='Claude'), bitrate=1800)
#         anim.save(output_video, writer=writer)
        
#         print(f"\nVideo created successfully: {output_video}")
        
#         # Display the animation in the notebook.
#         display(HTML(anim.to_jshtml()))
        
#     except Exception as e:
#         print(f"Error creating video: {e}")
#         print("\nYou can install imageio with: pip install imageio[ffmpeg]")
#         print("Or use the frames manually from the video_frames directory")

# %%
## Display Video in Notebook (Optional)

from IPython.display import Video

# Display the video in the notebook.
try:
    display(Video("joint_entropy_animation.mp4", embed=True, width=800))
    print("Video displayed above")
except Exception as e:
    print(f"Could not display video: {e}")
    print("You can find the video file: joint_entropy_animation.mp4")
