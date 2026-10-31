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
# # Description
#
# - Teach the `gymnasium.spaces` API by exploring every `Space` class
#   with small working example
# - Focus on primitives: what they represent, how they are constructed, what
#   attributes they hold, and how they compose
#
# - References:
#   - API: https://gymnasium.farama.org/api/spaces
#   - GitHub: https://github.com/Farama-Foundation/Gymnasium

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.
import numpy as np
import pandas as pd

# %%
import helpers.hdbg as hdbg
import helpers.hintrospection as hintros

_LOG = logging.getLogger(__name__)

hdbg.init_logger(verbosity=logging.INFO)

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library overview
#
# - **What problem it solves**: every RL environment must declare the valid type
#   of its observations and actions. `gymnasium.spaces` provides a typed contract
#   system for this
#
# - **Key abstraction**: `gymnasium.spaces.Space`: a base class every space
#   implements. It exposes two core methods (`sample`, `contains`) and two core
#   attributes (`shape`, `dtype`)
#
# - **Mental model**:
#   ```
#   space.sample()        -> one valid random value
#   space.contains(x)     -> True if x is valid in this space
#   x in space            -> equivalent to space.contains(x)
#   ```
#
# - **Key classes**:
#   - `Discrete(n)`: integers 0...n-1 (or start...start+n-1)
#   - `Box(low, high, shape)`: continuous tensor with per-element bounds
#   - `MultiBinary(n)`: binary array of shape n
#   - `MultiDiscrete(nvec)`: vector of independent discrete variables
#   - `Text(max_length)`: strings from a charset
#   - `Dict(spaces)`: named collection of spaces
#   - `Tuple(spaces)`: ordered collection of spaces
#   - `Sequence(space)`: variable-length sequence of one space
#   - `Graph(node_space, edge_space)`: graph with node/edge features
#   - `OneOf(spaces)`: exclusive union (exactly one constituent space per sample)

# %% [markdown]
# # Part 1: The base `Space` class

# %% [markdown]
# ## Cell 1.1: Inspect the `Space` base class
#
# - `gymnasium.spaces.Space` is the abstract base class for all spaces
# - It defines the interface: `sample()`, `contains()`, `seed()`, `shape`,
#   `dtype`, `np_random`
# - Every concrete space subclasses it

# %%
from gymnasium import spaces

# Inspect the public interface of the base Space class.
hintros.print_obj_info(spaces.Space)

# %%
# Confirm that all concrete spaces are subclasses of Space.
space_classes = [
    spaces.Discrete,
    spaces.Box,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
    spaces.Text,
    spaces.Dict,
    spaces.Tuple,
    spaces.Sequence,
    spaces.Graph,
    spaces.OneOf,
]
rows = [
    {
        "class": cls.__name__,
        "is_subclass_of_Space": issubclass(cls, spaces.Space),
    }
    for cls in space_classes
]
display(pd.DataFrame(rows))

# %% [markdown]
# # Part 2: Fundamental spaces

# %% [markdown]
# ## Cell 2.1: `Discrete(n)`: integers 0...n-1
#
# - Represents the set $\{start, start+1, \ldots, start+n-1\}$
# - Default `start=0`, so the set is $\{0, 1, \ldots, n-1\}$
# - Typical use: discrete action spaces (turn left / right / stay)

# %%
# Construct the simplest Discrete space.
d = spaces.Discrete(4)
# Seed the space so that reruns print the same samples, see Cell 4.1.
d.seed(1)
print("d=", d)
print("type(d)=", type(d))

# %%
# Inspect core attributes.
print("d.n=", d.n)
print("d.start=", d.start)
print("d.dtype=", d.dtype)
print("d.shape=", d.shape)

# %%
# Sample several random values.
samples = [int(d.sample()) for _ in range(10)]
print("samples=", samples)

# %%
# Membership check: contains() and `in` operator are equivalent.
print("d.contains(2)=", d.contains(2))
print("2 in d=", 2 in d)
print("d.contains(5)=", d.contains(5))

# %%
# Non-zero start: set becomes {10, 11, 12}.
d_offset = spaces.Discrete(3, start=10)
d_offset.seed(0)
print("d_offset=", d_offset)
samples_offset = [int(d_offset.sample()) for _ in range(8)]
print("samples_offset=", samples_offset)

# %%
# Masked sampling: only allow actions 0 and 2, not 1 or 3.
mask = np.array([1, 0, 1, 0], dtype=np.int8)
masked_samples = [int(d.sample(mask=mask)) for _ in range(10)]
print("masked_samples (only 0 or 2)=", masked_samples)

# %% [markdown]
# ## Cell 2.2: `Box(low, high, shape)`: continuous tensor
#
# - Represents a bounded or unbounded real-valued tensor
# - Each element has its own `[low, high]` bound
# - Typical use: continuous observation spaces (position, velocity, angle)

# %%
# Construct a 1-D Box with uniform bounds.
b = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
b.seed(0)
print("b=", b)
print("type(b)=", type(b))

# %%
# Inspect core attributes.
print("b.low=", b.low)
print("b.high=", b.high)
print("b.shape=", b.shape)
print("b.dtype=", b.dtype)

# %%
# Sample a random value.
sample = b.sample()
print("sample=", sample)
print("type(sample)=", type(sample))
print("sample.shape=", sample.shape)

# %%
# Membership check.
inside = np.array([0.5, -0.5, 0.0], dtype=np.float32)
outside = np.array([2.0, 0.0, 0.0], dtype=np.float32)
print("inside in b=", b.contains(inside))
print("outside in b=", b.contains(outside))

# %%
# Per-element bounds: each dimension can have a different range.
b_per_elem = spaces.Box(
    low=np.array([0.0, -10.0, -np.inf]),
    high=np.array([1.0, 10.0, np.inf]),
    dtype=np.float64,
)
print("b_per_elem.low=", b_per_elem.low)
print("b_per_elem.high=", b_per_elem.high)

# %%
# is_bounded() checks whether all dimensions have finite bounds.
print("b.is_bounded()=", b.is_bounded())
print("b_per_elem.is_bounded()=", b_per_elem.is_bounded())

# %%
# 2-D Box: shape (2, 3) - useful for image-like observations.
b_2d = spaces.Box(low=0, high=255, shape=(2, 3), dtype=np.uint8)
b_2d.seed(0)
print("b_2d.shape=", b_2d.shape)
print("b_2d.sample()=")
print(b_2d.sample())

# %% [markdown]
# ## Cell 2.3: `MultiBinary(n)`: binary array
#
# - Represents an array of independent binary (0/1) variables
# - Each element is sampled as an independent fair coin toss by default
# - Typical use: multi-label action spaces (press key A and B simultaneously)

# %%
# Construct a flat MultiBinary space of 5 bits.
mb = spaces.MultiBinary(5)
mb.seed(0)
print("mb=", mb)
print("mb.n=", mb.n)
print("mb.shape=", mb.shape)

# %%
# Sample: each element is independently 0 or 1.
samples = [list(map(int, mb.sample())) for _ in range(5)]
print("samples=")
display(pd.DataFrame(samples, columns=[f"bit{i}" for i in range(5)]))

# %%
# Mask: 0=force 0, 1=force 1, 2=random.
# Bits 0 and 4 are forced to 0 and 1; bits 1-3 are random.
mask = np.array([0, 2, 2, 2, 1], dtype=np.int8)
masked = mb.sample(mask=mask)
print("masked_sample=", list(map(int, masked)))

# %%
# 2-D MultiBinary: shape (2, 3) grid of binary values.
mb_2d = spaces.MultiBinary([2, 3])
mb_2d.seed(0)
print("mb_2d.n=", mb_2d.n)
print("mb_2d.shape=", mb_2d.shape)
print("mb_2d.sample()=")
print(mb_2d.sample())

# %% [markdown]
# ## Cell 2.4: `MultiDiscrete(nvec)`: vector of independent discrete variables
#
# - Represents a Cartesian product of discrete spaces
# - `nvec[i]` is the number of values for dimension $i$
# - Typical use: game controller with multiple independent axes

# %%
# 3 independent dimensions: first has 3 values, second has 4, third has 2.
md = spaces.MultiDiscrete([3, 4, 2])
md.seed(0)
print("md=", md)
print("md.nvec=", md.nvec)
print("md.shape=", md.shape)

# %%
# Each sample is a vector of independent integers.
samples = [list(map(int, md.sample())) for _ in range(6)]
display(
    pd.DataFrame(samples, columns=["dim0 (0-2)", "dim1 (0-3)", "dim2 (0-1)"])
)

# %%
# Non-zero start: each dimension starts at a different offset.
md_offset = spaces.MultiDiscrete([3, 4], start=[10, 20])
md_offset.seed(0)
print("md_offset.nvec=", md_offset.nvec)
print("md_offset.start=", md_offset.start)
samples_offset = [list(map(int, md_offset.sample())) for _ in range(5)]
display(pd.DataFrame(samples_offset, columns=["dim0 (10-12)", "dim1 (20-23)"]))

# %%
# Membership check.
valid = np.array([2, 3, 1])
invalid = np.array([3, 3, 1])  # dim0 only goes up to 2.
print("valid in md=", md.contains(valid))
print("invalid in md=", md.contains(invalid))

# %% [markdown]
# ## Cell 2.5: `Text(max_length)`: variable-length strings
#
# - Represents strings of characters from a specified charset
# - Length is bounded by `[min_length, max_length]`
# - Typical use: natural language action spaces, instruction following

# %%
# Default charset: alphanumeric (a-z, A-Z, 0-9).
t = spaces.Text(max_length=8)
t.seed(0)
print("t=", t)
print("t.max_length=", t.max_length)
print("t.min_length=", t.min_length)
print("len(t.characters)=", len(t.characters))

# %%
# Sample several strings.
samples = [t.sample() for _ in range(6)]
print("samples=", samples)

# %%
# Custom charset: only lowercase vowels.
t_vowels = spaces.Text(max_length=5, min_length=2, charset="aeiou")
t_vowels.seed(0)
print("t_vowels charset=", t_vowels.characters)
vowel_samples = [t_vowels.sample() for _ in range(6)]
print("vowel_samples=", vowel_samples)

# %%
# Membership check.
print('t.contains("Hello")=', t.contains("Hello"))
# String with non-alphanumeric character fails.
print('t.contains("Hi!")=', t.contains("Hi!"))
# String too long fails.
print('t.contains("TooLongStr")=', t.contains("TooLongStr"))

# %% [markdown]
# # Part 3: Composite spaces

# %% [markdown]
# ## Cell 3.1: `Dict(spaces)`: named collection of spaces
#
# - Represents an ordered dictionary of heterogeneous spaces
# - A sample is a `dict` with one value per key
# - Typical use: structured observations (image + scalar sensors)

# %%
# Construct a Dict with two subspaces.
ds = spaces.Dict(
    {
        "pos": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        "speed": spaces.Discrete(5),
    }
)
ds.seed(0)
print("ds=", ds)

# %%
# Inspect the subspaces dictionary.
for key, space in ds.spaces.items():
    print(f"ds.spaces[{key!r}]=", space)

# %%
# A sample is a dict with one value per key.
sample = ds.sample()
print("type(sample)=", type(sample))
print("sample['pos']=", sample["pos"])
print("sample['speed']=", sample["speed"])

# %%
# Dict can also be constructed with keyword arguments.
ds_kw = spaces.Dict(
    sensor=spaces.Box(low=-1.0, high=1.0, shape=(3,)),
    flag=spaces.MultiBinary(2),
)
print("ds_kw.spaces.keys()=", list(ds_kw.spaces.keys()))

# %%
# Membership check: each subspace's contains() must pass.
valid_sample = {"pos": np.array([0.5, 0.5], dtype=np.float32), "speed": 3}
invalid_sample = {"pos": np.array([0.5, 0.5], dtype=np.float32), "speed": 10}
print("valid_sample in ds=", ds.contains(valid_sample))
print("invalid_sample in ds=", ds.contains(invalid_sample))

# %% [markdown]
# ## Cell 3.2: `Tuple(spaces)`: ordered collection of spaces
#
# - Represents a fixed-length heterogeneous tuple of spaces
# - A sample is a Python `tuple` with one value per subspace
# - Similar to `Dict` but accessed by index rather than key

# %%
# Construct a Tuple of (Discrete, Box).
tup = spaces.Tuple(
    (
        spaces.Discrete(3),
        spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
    )
)
tup.seed(0)
print("tup=", tup)
print("len(tup.spaces)=", len(tup.spaces))

# %%
# Sample: returns a Python tuple.
sample = tup.sample()
print("type(sample)=", type(sample))
print("sample[0] (Discrete)=", sample[0])
print("sample[1] (Box)=", sample[1])

# %%
# Masked sampling per subspace.
# Only allow action 1 in Discrete(3); Box samples freely.
mask_discrete = np.array([0, 1, 0], dtype=np.int8)
masked_sample = tup.sample(mask=(mask_discrete, None))
print("masked_sample[0] (only 1)=", masked_sample[0])

# %% [markdown]
# ## Cell 3.3: `Sequence(space)`: variable-length sequence
#
# - Represents variable-length sequences where each element belongs to one subspace
# - Length is sampled from a geometric distribution by default
# - Typical use: lists of objects of unknown count (agents, obstacles)

# %%
# Each element is a Discrete(4); length varies per sample.
seq = spaces.Sequence(spaces.Discrete(4))
seq.seed(0)
print("seq=", seq)
print("seq.feature_space=", seq.feature_space)

# %%
# Each sample is a Python tuple of varying length.
for _ in range(5):
    s = seq.sample()
    print(f"  len={len(s)}  values={s}")

# %%
# Fixed-length mask: force exactly 3 elements.
fixed_samples = [seq.sample(mask=(3, None)) for _ in range(4)]
print("fixed_length=3 samples=")
for s in fixed_samples:
    print(" ", s)

# %%
# stack=True: stacks the tuple elements into a single numpy array.
seq_stacked = spaces.Sequence(
    spaces.Box(low=0.0, high=1.0, shape=(2,)), stack=True
)
s_stacked = seq_stacked.sample(mask=(4, None))
print("type(s_stacked)=", type(s_stacked))
print("s_stacked.shape=", s_stacked.shape)

# %% [markdown]
# ## Cell 3.4: `Graph(node_space, edge_space)`: graphs with node/edge features
#
# - Represents a directed graph with node and edge feature spaces
# - A sample is a `GraphInstance` with `.nodes`, `.edges`, and `.edge_links`
# - Typical use: molecule generation, social network observations

# %%
# Node features: 3-D continuous vectors. Edge features: Discrete(5) labels.
g = spaces.Graph(
    node_space=spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
    edge_space=spaces.Discrete(5),
)
g.seed(0)
print("g=", g)
print("g.node_space=", g.node_space)
print("g.edge_space=", g.edge_space)

# %%
# Sample a graph with exactly 4 nodes and 3 edges.
gi = g.sample(num_nodes=4, num_edges=3)
print("type(gi)=", type(gi))
print("gi.nodes.shape=", gi.nodes.shape)  # (4, 3): 4 nodes, 3 features each.
print("gi.edge_links.shape=", gi.edge_links.shape)  # (3, 2): 3 edges, src/dst.

# %%
# Inspect node features and edge links.
print("gi.nodes (first 2)=")
print(gi.nodes[:2])
print("gi.edges=", gi.edges)  # Edge feature per edge.
print("gi.edge_links=")  # Shape (num_edges, 2): [src, dst].
print(gi.edge_links)

# %% [markdown]
# ## Cell 3.5: `OneOf(spaces)`: exclusive union
#
# - Represents a direct sum of spaces: a sample belongs to exactly one constituent
# - A sample is `(index, value)` where `index` identifies which subspace was sampled
# - Typical use: multi-modal action spaces (text command OR array of positions)

# %%
# Union of Discrete(3) and Box(shape=(2,)).
oo = spaces.OneOf(
    (
        spaces.Discrete(3),
        spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
    )
)
oo.seed(0)
print("oo=", oo)
print("len(oo.spaces)=", len(oo.spaces))

# %%
# Each sample is (space_index, sampled_value).
for _ in range(6):
    idx, val = oo.sample()
    print(f"  idx={idx}  val={val}")

# %% [markdown]
# # Part 4: Common API patterns

# %% [markdown]
# ## Cell 4.1: Seeding for reproducibility
#
# - Every space has a PRNG (`np_random`) used for sampling
# - `space.seed(n)` seeds it; subsequent `space.sample()` calls are deterministic
# - Exception: `Text` with its default charset also depends on `PYTHONHASHSEED`,
#   since that charset is a `frozenset`, whose order changes between processes
# - For composite spaces, `seed()` returns a dict/tuple of seed values for
#   each subspace

# %%
# Same seed -> same samples for Discrete.
d_seed = spaces.Discrete(10)
d_seed.seed(42)
s1 = [int(d_seed.sample()) for _ in range(5)]
d_seed.seed(42)
s2 = [int(d_seed.sample()) for _ in range(5)]
print("s1=", s1)
print("s2=", s2)
print("s1 == s2:", s1 == s2)

# %%
# Composite seeding: Dict.seed() returns a dict of seed values.
ds_seed = spaces.Dict(
    {"a": spaces.Discrete(4), "b": spaces.Box(0.0, 1.0, shape=(2,))}
)
seeds = ds_seed.seed(0)
print("seeds=", seeds)
print("type(seeds)=", type(seeds))

# %% [markdown]
# ## Cell 4.2: Flattening spaces
#
# - `spaces.utils.flatten(space, x)` converts any sample to a 1-D float array
# - `spaces.utils.flatten_space(space)` returns the equivalent `Box` space
# - `spaces.utils.flatdim(space)` returns the dimension of the flattened array
# - Useful for neural network inputs: the model sees a flat float vector regardless of
#   the original space structure

# %%
from gymnasium.spaces import utils as space_utils

# Flatten Discrete(4): one-hot encode into a 4-D float vector.
d_flat = spaces.Discrete(4)
d_flat.seed(0)
sample_d = d_flat.sample()
flat_d = space_utils.flatten(d_flat, sample_d)
print("sample_d=", sample_d)
print("flat_d=", flat_d)
print("space_utils.flatdim(d_flat)=", space_utils.flatdim(d_flat))

# %%
# Flatten a Box: already flat, returned as-is.
b_flat = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
b_flat.seed(0)
sample_b = b_flat.sample()
flat_b = space_utils.flatten(b_flat, sample_b)
print("sample_b=", sample_b)
print("flat_b=", flat_b)

# %%
# Flatten a Dict: subspaces are concatenated in key order.
ds_flat = spaces.Dict(
    {
        "a": spaces.Discrete(3),
        "b": spaces.Box(0.0, 1.0, shape=(2,)),
    }
)
ds_flat.seed(0)
sample_ds = ds_flat.sample()
flat_ds = space_utils.flatten(ds_flat, sample_ds)
print("sample_ds=", sample_ds)
print("flat_ds=", flat_ds)
print("flatdim(ds_flat)=", space_utils.flatdim(ds_flat))  # 3 (one-hot) + 2 = 5.

# %%
# flatten_space() returns the equivalent Box.
flat_space = space_utils.flatten_space(ds_flat)
print("flat_space=", flat_space)
print("flat_space.shape=", flat_space.shape)

# %%
# unflatten() is the inverse of flatten().
roundtrip = space_utils.unflatten(ds_flat, flat_ds)
print("roundtrip['a']=", roundtrip["a"])
print("roundtrip['b']=", roundtrip["b"])

# %% [markdown]
# ## Cell 4.3: Comparing all fundamental spaces side-by-side
#
# - Quick comparison of key attributes across all five fundamental space types
# - Useful reference when choosing the right space for an observation or action

# %%
# Build a comparison table for all fundamental space types.
specs = [
    ("Discrete(4)", spaces.Discrete(4)),
    ("Box(0,1,shape=(3,))", spaces.Box(0, 1, shape=(3,), dtype=np.float32)),
    ("MultiBinary(4)", spaces.MultiBinary(4)),
    ("MultiDiscrete([3,4,2])", spaces.MultiDiscrete([3, 4, 2])),
    ("Text(max_length=5)", spaces.Text(max_length=5)),
]
rows = []
for name, sp in specs:
    sp.seed(0)
    rows.append(
        {
            "space": name,
            "shape": str(sp.shape),
            "dtype": str(getattr(sp, "dtype", "N/A")),
            "sample": str(sp.sample()),
            "flatdim": str(space_utils.flatdim(sp)),
        }
    )
display(pd.DataFrame(rows))

# %% [markdown]
# # Part 5: Interactive exploration

# %% [markdown]
# ## Cell 5.1: Explore any space's interface
#
# - Use `dir()` to see all available attributes and methods
# - Every space exposes the same base interface regardless of type

# %%
# All public attributes shared across any Space.
d_explore = spaces.Discrete(3)
public = [a for a in dir(d_explore) if not a.startswith("_")]
print("public attributes of Discrete(3)=", public)

# %%
# Explore: what extra attributes does Box have that Discrete doesn't?
b_explore = spaces.Box(0.0, 1.0, shape=(3,))
extra_in_box = set(dir(b_explore)) - set(dir(d_explore))
print("extra_in_box=", sorted(extra_in_box))

# %%
# What does np_random look like?
d_explore.seed(99)
print("type(d_explore.np_random)=", type(d_explore.np_random))
# Use it directly to generate non-space random numbers.
print("random float from np_random=", d_explore.np_random.random())

# %%
# What happens if you nest Dict inside Dict?
nested = spaces.Dict(
    {
        "inner": spaces.Dict(
            {
                "x": spaces.Discrete(2),
                "y": spaces.Box(0.0, 1.0, shape=(1,)),
            }
        ),
        "flag": spaces.MultiBinary(3),
    }
)
nested.seed(0)
nested_sample = nested.sample()
print("nested_sample['inner']['x']=", nested_sample["inner"]["x"])
print("nested_sample['inner']['y']=", nested_sample["inner"]["y"])
print("nested_sample['flag']=", nested_sample["flag"])

# %% [markdown]
# # Summary: the mental model
#
# - A `Space` is a **typed contract**: it declares what values are valid and can
#   sample random valid values via `sample()` and check membership via `contains()`
#
# - **Fundamental spaces** cover common primitives:
#   - `Discrete(n)`: integers (categorical actions)
#   - `Box(low, high, shape)`: real-valued tensors (continuous observations)
#   - `MultiBinary(n)`: binary arrays (multi-label flags)
#   - `MultiDiscrete(nvec)`: vector of independent categoricals (multi-axis controllers)
#   - `Text(max_length)`: variable-length strings (NLP interfaces)
#
# - **Composite spaces** build richer structures from primitives:
#   - `Dict` / `Tuple`: fixed-structure heterogeneous collections (by key or index)
#   - `Sequence(space)`: variable-length list of homogeneous elements
#   - `Graph(node_space, edge_space)`: graphs with typed node/edge features
#   - `OneOf(spaces)`: exclusive union, sample returns `(index, value)`
#
# - All spaces share the same interface (`sample`, `contains`, `seed`, `flatten`),
#   so RL algorithms can treat them uniformly regardless of their internal structure
