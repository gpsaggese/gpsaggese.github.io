"""
The recording agent and the four operations: `record_run`, `fork`,
`run_to_end`, and the store's `snapshots`.

The agent is mini-swe-agent's `DefaultAgent` with two recording hooks:
- Hook 1 (model call): before the call, the hash of the exact API messages, the
  model id and the sampling settings; after it, the reply text, finish reason
  and token usage.
- Hook 2 (command): the command, its output, exit code and duration.

A checkpoint is taken at every quiet moment: after the step's observation (or
format-error message) has been appended and before the next model call, when
the conversation and the board agree.

Interventions replace a model call with a reply taken from a tape (`replay`,
the control group) or given by the experimenter (`force`, a planted reply); the
replacement is recorded on the tape with its source.

Restore rules: every fork gets a fresh environment built from the saved one and
a new branch id that records its parent checkpoint and intervention. A fork is
never made by "resuming" an old branch.

Import as:

import caaf_agent as cagent
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from minisweagent import __version__ as MINI_VERSION
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
)
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.actions_text import parse_regex_actions

import caaf_chess as cchess
import caaf_store as cstore

ROOT = Path(__file__).resolve().parent

# Harness metadata that changes between identical runs; the raw response is kept
# on the tape.
_VOLATILE_EXTRA = ("timestamp", "response")


def load_config(name: str = "") -> Dict:
    name = name or os.environ.get("CAAF_CONFIG", "chess_haiku.yaml")
    return yaml.safe_load((ROOT / "config" / name).read_text())


def api_messages(messages: List[Dict]) -> List[Dict]:
    """
    Return what the model receives (mini-swe-agent strips `extra` first).
    """
    return [{k: v for k, v in m.items() if k != "extra"} for m in messages]


def _canonical_assistant(message: Dict) -> Dict:
    """
    Keep only role and content in the conversation.

    litellm adds provider fields that are null or provider-specific; they are
    moved to `extra` (and the raw response is on the tape). A reply replayed
    from the tape is then exactly the message the original run sent back.
    """
    extra = dict(message.get("extra", {}))
    dropped = {
        k: v for k, v in message.items() if k not in ("role", "content", "extra")
    }
    if dropped:
        extra["api_fields_dropped"] = dropped
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "extra": extra,
    }


def _stable_messages(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        m = dict(m)
        if "extra" in m:
            m["extra"] = {
                k: v for k, v in m["extra"].items() if k not in _VOLATILE_EXTRA
            }
        out.append(m)
    return out


# #############################################################################
# CaafModel
# #############################################################################


class CaafModel(LitellmTextbasedModel):
    """
    mini-swe-agent's text-based litellm model with the spending ledger: checked
    before, charged after every call.
    """

    budget: Optional[cstore.Ledger] = None

    def query(self, messages: List[Dict], **kwargs) -> Dict:
        if self.budget is not None:
            try:
                self.budget.check()
            except cstore.BudgetExceeded as e:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": f"BudgetExceeded: {e}",
                        "extra": {
                            "exit_status": "BudgetExceeded",
                            "submission": "",
                        },
                    }
                )
        try:
            message = super().query(messages, **kwargs)
        except FormatError as e:
            raw = e.messages[0].get("extra", {}).get("response")
            if self.budget is not None and isinstance(raw, dict):
                e.messages[0]["extra"]["cost"] = self.budget.add(
                    raw.get("usage"), self.config.model_name
                )
            raise
        if self.budget is not None:
            usage = message["extra"]["response"].get("usage")
            message["extra"]["cost"] = self.budget.add(
                usage, self.config.model_name
            )
        return message


def make_model(model_cfg: Dict) -> CaafModel:
    keys = ("observation_template", "format_error_template", "set_cache_control")
    extra = {k: model_cfg[k] for k in keys if k in model_cfg}
    m = CaafModel(
        model_name=model_cfg["model_name"],
        model_kwargs=dict(model_cfg["model_kwargs"]),
        cost_tracking="ignore_errors",
        **extra,
    )
    if model_cfg.get("budget"):
        b = model_cfg["budget"]
        m.budget = cstore.Ledger(ROOT / b["ledger"], b["cap_usd"], b["prices"])
    return m


# #############################################################################
# RecordingAgent
# #############################################################################


class RecordingAgent(DefaultAgent):
    def __init__(
        self,
        model: CaafModel,
        env: cchess.ChessEnv,
        *,
        store: cstore.Store,
        run_id: str,
        branch_id: str,
        overrides: Optional[Dict] = None,
        harness: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, env, **kwargs)
        self.store, self.run_id, self.branch_id = store, run_id, branch_id
        # Call index -> {"content": str, "source": str}.
        self.overrides = overrides or {}
        self.harness = harness or {}
        self.tape = cstore.Tape(store, run_id, branch_id)
        self.last_snapshot_id: Optional[str] = None
        self.snapshot_seq = 0
        # `run_to_end` can stop early (replay checks).
        self.stop_after_calls: Optional[int] = None

    # ---- hook 1: the model call

    def query(self) -> Dict:
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        k = self.n_calls
        sent = api_messages(self.messages)
        request = {
            "input_hash": cstore.sha256(cstore.canonical(sent)),
            "n_messages": len(sent),
            "model": self.model.config.model_name,
            "sampling": self.sampling(),
        }
        override = self.overrides.get(k)
        t0 = time.monotonic()
        try:
            if override:
                message = self._override_message(override["content"])
                response = {
                    "content": override["content"],
                    "finish_reason": "override",
                    "usage": None,
                    "source": override["source"],
                }
            else:
                message = _canonical_assistant(self.model.query(self.messages))
                raw = message["extra"]["response"]
                choice = raw["choices"][0]
                response = {
                    "content": choice["message"].get("content") or "",
                    "finish_reason": choice["finish_reason"],
                    "usage": raw.get("usage"),
                    "source": "model",
                }
        except FormatError as e:
            extra = e.messages[0]["extra"]
            raw = extra.get("response") or {}
            if override:
                content = override["content"]
            elif isinstance(raw, dict) and raw.get("choices"):
                content = raw["choices"][0]["message"].get("content") or ""
            else:
                content = extra.get("model_response", "")
            ok = isinstance(raw, dict) and raw.get("choices")
            self.tape.record(
                "model_call",
                k,
                request,
                {
                    "content": content,
                    "finish_reason": raw["choices"][0].get("finish_reason")
                    if ok
                    else None,
                    "usage": raw.get("usage") if isinstance(raw, dict) else None,
                    "source": override["source"] if override else "model",
                    "format_error": True,
                },
                {"seconds": round(time.monotonic() - t0, 3)},
            )
            raise
        self.tape.record(
            "model_call",
            k,
            request,
            response,
            {"seconds": round(time.monotonic() - t0, 3)},
        )
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def _override_message(self, content: str) -> Dict:
        """
        Build the same canonical message the model's reply would have produced.
        """
        actions = parse_regex_actions(
            content,
            action_regex=self.model.config.action_regex,
            format_error_template=self.model.config.format_error_template,
            template_kwargs={"finish_reason": "override"},
        )
        extra = {"actions": actions, "cost": 0.0, "override": True}
        return {"role": "assistant", "content": content, "extra": extra}

    def sampling(self) -> Dict:
        kw = dict(self.model.config.model_kwargs)
        kw.pop("api_base", None)
        kw.pop("api_key", None)
        return kw

    # ---- hook 2: the command

    def execute_actions(self, message: Dict) -> List[Dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            out = self.env.execute(action)
            outputs.append(out)
            self.tape.record(
                "command",
                self.n_calls,
                {"command": action["command"]},
                {
                    "output": out.get("output", ""),
                    "returncode": out.get("returncode"),
                    "exception_info": out.get("exception_info", ""),
                },
                {"seconds": round(self.env.last_duration, 3)},
            )
        added = self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )
        for o in outputs:
            for line in o.get("output", "").splitlines():
                if line.startswith(cchess.GAME_OVER):
                    raise Submitted(
                        {
                            "role": "exit",
                            "content": line,
                            "extra": {
                                "exit_status": "GameOver",
                                "submission": line,
                            },
                        }
                    )
        return added

    # ---- the quiet moment

    def context(self) -> Dict:
        return {
            "schema": "caaf.context/1",
            "messages": _stable_messages(self.messages),
            "counters": {
                "n_calls": self.n_calls,
                "cost": self.cost,
                "step_limit": self.config.step_limit,
                "n_consecutive_format_errors": self.n_consecutive_format_errors,
            },
            "model": {
                "name": self.model.config.model_name,
                "sampling": self.sampling(),
                "seed_policy": "none (the API takes no seed)",
            },
            "agent_config": self.config.model_dump(mode="json"),
            "template_vars": self.extra_template_vars,
            "harness": {**self.harness, "mini_swe_agent": MINI_VERSION},
        }

    def take_snapshot(self) -> Dict:
        snap = self.store.put_snapshot(
            run_id=self.run_id,
            branch_id=self.branch_id,
            seq=self.snapshot_seq,
            step=self.n_calls,
            parent_id=self.last_snapshot_id,
            context=self.context(),
            env_state=self.env.state(),
        )
        self.last_snapshot_id = snap["id"]
        self.snapshot_seq += 1
        return snap

    def restore_context(self, ctx: Dict) -> None:
        self.messages = [dict(m) for m in ctx["messages"]]
        c = ctx["counters"]
        self.n_calls, self.cost = c["n_calls"], c["cost"]
        self.n_consecutive_format_errors = c["n_consecutive_format_errors"]
        self.extra_template_vars = dict(ctx["template_vars"])

    # ---- loop shared by fresh runs and forks

    def start(self, task: str) -> Dict:
        self.extra_template_vars |= {"task": task}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render_template(self.config.system_template),
            ),
            self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            ),
        )
        return self.take_snapshot()

    def loop(self) -> Dict:
        while self.messages[-1].get("role") != "exit":
            if (
                self.stop_after_calls is not None
                and self.n_calls >= self.stop_after_calls
            ):
                break
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                limit = self.config.max_consecutive_format_errors
                if 0 < limit <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {
                                "exit_status": "RepeatedFormatError",
                                "submission": "",
                            },
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            self.take_snapshot()
        self.tape.close()
        if self.messages[-1].get("role") == "exit":
            return self.messages[-1].get("extra", {})
        return {"exit_status": "Paused"}


# #############################################################################
# The four operations
# #############################################################################


@dataclass
class Branch:
    store: cstore.Store
    agent: RecordingAgent
    env: cchess.ChessEnv
    run_id: str
    branch_id: str
    info: Dict = field(default_factory=dict)


def harness_info(config: Dict) -> Dict:
    def git(*args: str) -> str:
        r = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True
        )
        return r.stdout.strip()

    tb = {
        p.name: cstore.sha256(p.read_bytes())
        for p in sorted((ROOT / "data" / "syzygy").glob("KRvK.rtb?"))
    }
    return {
        "harness_commit": git("rev-parse", "HEAD") or None,
        "harness_dirty": bool(git("status", "--porcelain", "--", ".")),
        "tablebase_sha256": tb,
        "config_name": config.get("name", ""),
    }


def step_limit_for(start_fen: str, config: Dict) -> int:
    """
    Return White's moves left before the fifty-move rule, plus some spare steps.
    """
    clock = int(start_fen.split()[4])
    return (100 - clock + 1) // 2 + config["extra_steps"]


def _agent(
    store, env, run_id, branch_id, model_cfg, agent_cfg, harness, overrides
):
    return RecordingAgent(
        make_model(model_cfg),
        env,
        store=store,
        run_id=run_id,
        branch_id=branch_id,
        overrides=overrides,
        harness=harness,
        **agent_cfg,
    )


def record_run(
    store: cstore.Store,
    start_fen: str,
    *,
    config: Dict,
    label: str = "",
    overrides: Optional[Dict] = None,
) -> Branch:
    """
    Start a new game (root branch) and take checkpoint 0.

    Call `run_to_end` to play it.
    """
    harness = harness_info(config)
    run_id = store.new_run(
        {
            "label": label,
            "start_fen": start_fen,
            "config": config,
            "harness": harness,
        }
    )
    branch_id = store.new_branch(run_id, intervention={"type": "root"})
    env = cchess.ChessEnv(start_fen)
    board = env.execute({"command": "play show"})["output"]
    agent_cfg = dict(config["agent"])
    agent_cfg["step_limit"] = step_limit_for(start_fen, config)
    agent = _agent(
        store,
        env,
        run_id,
        branch_id,
        config["model"],
        agent_cfg,
        harness,
        overrides,
    )
    snap0 = agent.start(
        task=config["task_template"].strip() + "\n\n" + board.strip()
    )
    return Branch(
        store, agent, env, run_id, branch_id, {"snapshot0": snap0["id"]}
    )


def fork(
    store: cstore.Store,
    snapshot_id: str,
    *,
    config: Dict,
    intervention: Optional[Dict] = None,
) -> Branch:
    """
    Start a new branch from a saved checkpoint, with verified env and context.

    intervention:
      {"type": "free"}
          the model decides every later step (re-decide)
      {"type": "replay", "source_branch": b, "steps": [k, ...]}
          step k's reply comes from b's tape (control group, replay checks)
      {"type": "force", "step": k, "content": "..."}
          planted reply at step k
    """
    intervention = intervention or {"type": "free"}
    snap = store.snapshot(snapshot_id)
    ctx = json.loads(store.get_blob("context", snap["context_hash"]))
    env_state = store.get_blob("env", snap["env_hash"])
    overrides = {}
    if intervention["type"] == "replay":
        src = intervention["source_branch"]
        calls = {
            e["step"]: e
            for e in store.read_tape(src)
            if e["kind"] == "model_call"
        }
        for k in intervention["steps"]:
            overrides[k] = {
                "content": calls[k]["response"]["content"],
                "source": f"tape:{src}:{k}",
            }
    elif intervention["type"] == "force":
        overrides[intervention["step"]] = {
            "content": intervention["content"],
            "source": "forced",
        }
    branch_id = store.new_branch(
        snap["run_id"],
        parent_snapshot_id=snapshot_id,
        fork_step=snap["step"],
        intervention=intervention,
    )
    env = cchess.ChessEnv(json.loads(env_state)["start_fen"])
    env.restore(env_state, snap["env_hash"])
    harness = {k: v for k, v in ctx["harness"].items() if k != "mini_swe_agent"}
    model_cfg = {
        **config["model"],
        "model_name": ctx["model"]["name"],
        "model_kwargs": {
            **config["model"]["model_kwargs"],
            **ctx["model"]["sampling"],
        },
    }
    agent = _agent(
        store,
        env,
        snap["run_id"],
        branch_id,
        model_cfg,
        ctx["agent_config"],
        harness,
        overrides,
    )
    agent.restore_context(ctx)
    agent.last_snapshot_id = snapshot_id
    if cstore.sha256(cstore.canonical(agent.context())) != snap["context_hash"]:
        raise RuntimeError("restored context does not match the checkpoint")
    info = {
        "parent": snapshot_id,
        "env_verified": True,
        "context_verified": True,
    }
    return Branch(store, agent, env, snap["run_id"], branch_id, info)


def run_to_end(
    branch: Branch, *, stop_after_calls: Optional[int] = None
) -> Dict:
    """
    Play until the game ends or a limit is hit; return the referee's outcome.
    """
    branch.agent.stop_after_calls = stop_after_calls
    exit_info = branch.agent.loop()
    game = branch.env.game
    ref = cchess.referee(game["start_fen"], game["moves_uci"])
    outcome = {
        "won": game["status"] == "checkmate",
        "status": game["status"],
        "exit_status": exit_info.get("exit_status"),
        "n_calls": branch.agent.n_calls,
        "cost_usd": round(branch.agent.cost, 6),
        "moves_san": game["moves_san"],
        "last_snapshot": branch.agent.last_snapshot_id,
        "pivot_white_move": ref["pivot_white_move"],
        "end": ref["end"],
    }
    status = "paused" if stop_after_calls else "finished"
    branch.store.finish_branch(branch.branch_id, status, outcome)
    return outcome
