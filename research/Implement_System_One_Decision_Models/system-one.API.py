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
# # System One Decision Models: API in 60 Minutes
#
# This notebook uses Liquid d1-3B through the typed `/v1/systemone` API.
# A decision call returns probabilities for named answers without generating
# a prose response. We will ask a yes/no question, choose among named options,
# rate an ordered scale, combine questions over one state, and pick an action.
#
# Start the pinned server using the instructions in `README.md` before running
# this notebook. The examples are demonstrations, not an accuracy study.

# %% [markdown]
# ## Imports and Setup

# %%
# %load_ext autoreload
# %autoreload 2

import logging

logging.basicConfig(level=logging.WARNING)

# %%
from IPython.display import display

import research.Implement_System_One_Decision_Models.system_one_API_utils as soau

client = soau.SystemOneClient()
print(client.url)

# %% [markdown]
# ## 1. One State, One Yes/No Decision
#
# A **state** is the evidence the model sees. A `noul` question asks for a
# binary decision; its answer is `P(true)`. The model estimates this number.
# We need labeled cases before interpreting it as calibrated probability.

# %%
state = "Customer: My card was charged twice. Please refund the second charge today."
refund_question = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly request a refund?",
    }
}
refund_result = client.system_one(state, refund_question)
display(refund_result)

# %%
refund_probability = refund_result["answers"]["refund_requested"]["noul"]
print(f"Estimated P(refund explicitly requested): {refund_probability:.3f}")

# %% [markdown]
# ## 2. A Choice Among Named Options
#
# `criteria` defines the answer space. Names such as `billing` are stable keys;
# descriptions explain their meaning to the model. The returned distribution
# contains a probability for every declared option.

# %%
route_question = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this customer request?",
        "criteria": {
            "billing": "Payments and refunds",
            "shipping": "Delivery and tracking",
            "technical": "Product or account malfunction",
        },
    }
}
route_result = client.system_one(state, route_question)
route_answer = route_result["answers"]["route"]
display(route_answer)

# %%
route_probabilities = soau.choice_probabilities(
    route_answer, list(route_question["route"]["criteria"])
)
display(soau.plot_probabilities(route_probabilities, "Where should this ticket go?"))

# %% [markdown]
# `choice` is the most probable option. `confidence` is a separate API field
# measuring how concentrated the distribution is relative to uniform. Neither
# field says which action has the lowest business cost.

# %%
print("choice:", route_answer["choice"])
print("confidence:", round(route_answer["confidence"], 3))
print("top option probability:", round(max(route_probabilities.values()), 3))

# %% [markdown]
# ## 3. An Ordered Score
#
# A `score` question supplies 2–10 levels from low to high. The API returns
# a probability per level, a `legend`, and the probability-weighted expected
# level index. The expected index may lie between two levels.

# %%
urgency_question = {
    "urgency": {
        "type": "score",
        "instructions": "How urgently does this need a reply?",
        "criteria": ["can wait", "this week", "today", "immediately"],
    }
}
urgency_result = client.system_one(state, urgency_question)
urgency_answer = urgency_result["answers"]["urgency"]
display(urgency_answer)

# %%
urgency_probabilities = {
    urgency_answer["legend"][index]: probability
    for index, probability in urgency_answer["probabilities"].items()
}
display(soau.plot_probabilities(urgency_probabilities, "Reply urgency"))
print("Expected level index:", round(urgency_answer["score"], 3))

# %% [markdown]
# ## 4. Several Questions Over One State
#
# The request maps names to independent questions. A single response maps
# those same names to typed answers. This HTTP endpoint accepts one state per
# request; the model's native Python API also offers multi-state batching.

# %%
questions = {**refund_question, **route_question, **urgency_question}
combined_result = client.system_one(state, questions)
print("Answer names:", list(combined_result["answers"]))
print("Usage:", combined_result["usage"])

# %%
for name, answer in combined_result["answers"].items():
    print(name, answer)

# %% [markdown]
# ## 5. JSON State and a New Decision Schema
#
# A state can also be a JSON value. This new four-option question is created
# at inference time; there is no fixed label list in the client. Option names
# and descriptions define the decision for this request.

# %%
structured_state = {
    "channel": "email",
    "message": state,
    "priority": "standard",
}
next_step_question = {
    "next_step": {
        "type": "choice",
        "instructions": "What should the support agent do next?",
        "criteria": {
            "investigate_charge": "Check payment records",
            "request_tracking": "Ask for a parcel tracking number",
            "reset_password": "Send account reset instructions",
            "human_review": "Ask a human specialist to decide",
        },
    }
}
next_step_result = client.system_one(structured_state, next_step_question)
display(next_step_result["answers"]["next_step"])

# %% [markdown]
# ## 6. Turn a Distribution Into an Action
#
# A simple policy routes to the option with the highest predicted probability.
# That policy assumes every mistake costs the same. The application phase will
# make costs and a human-review action explicit.

# %%
route = soau.choose_top_option(route_probabilities)
print("Route selected by the top-probability policy:", route)

# %% [markdown]
# ## Try It
#
# Change the customer message, then rerun the cells. Try a shipping question,
# add a fourth route, or reverse the option order. Compare probabilities as
# well as the selected option. The saved Phase 1 probe in `probe_results.json`
# shows one such order change. A handful of examples cannot establish accuracy
# or calibration; those require labeled cases and a fixed evaluation protocol.
