"""
Freeze and grade the original 60-item controlled warm-up benchmark.

Import as:

import research.Noesis.monitor.monitor_warmup_benchmark as rnmonwaben
"""

import dataclasses
import hashlib
import json
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar


@dataclasses.dataclass(frozen=True)
class BenchmarkDefinition:
    """Store one original prompt and its private deterministic answer."""

    item_id: str
    family: str
    prompt: str
    grader_id: str
    expected: object


@dataclasses.dataclass(frozen=True)
class GraderConfig:
    """Store the answer material kept outside the provider call plan."""

    item_id: str
    grader_id: str
    expected: object


@dataclasses.dataclass(frozen=True)
class Grade:
    """Store one deterministic grade without altering raw evidence."""

    call_id: str
    item_id: str
    family: str
    route_name: str
    provider_status: str
    correct: Optional[bool]
    substantive_correct: Optional[bool]
    grader_id: str
    reason: str
    substantive_reason: str


def _answer_prompt(body: str) -> str:
    return body.strip() + "\n\nReturn only `ANSWER: <answer>`."


def _exact_prompt(body: str) -> str:
    return body.strip() + "\n\nReturn only the requested text, with no explanation."


DEFINITIONS: Tuple[BenchmarkDefinition, ...] = (
    # Math: original, deterministic short-answer problems.
    BenchmarkDefinition(
        "math-01", "math",
        _answer_prompt("Compute 37 × 18 − 245."),
        "integer-v1", 421,
    ),
    BenchmarkDefinition(
        "math-02", "math",
        _answer_prompt(
            "A fund is divided in the ratio 5:7 and totals $144. "
            "How many dollars are in the smaller share?"
        ),
        "integer-v1", 60,
    ),
    BenchmarkDefinition(
        "math-03", "math",
        _answer_prompt(
            "A $240 item is discounted by 15%, then taxed by 8% on the "
            "discounted price. Give the final price in cents."
        ),
        "integer-v1", 22032,
    ),
    BenchmarkDefinition(
        "math-04", "math",
        _answer_prompt("Solve for x: 3(2x − 5) + 4 = 5x + 12."),
        "integer-v1", 23,
    ),
    BenchmarkDefinition(
        "math-05", "math",
        _answer_prompt(
            "How many distinct 3-person committees can be selected from "
            "8 people?"
        ),
        "integer-v1", 56,
    ),
    BenchmarkDefinition(
        "math-06", "math",
        _answer_prompt(
            "A bag has 5 red and 3 blue balls. Two are drawn without "
            "replacement. What is the probability both are red? Give the "
            "fraction in lowest terms."
        ),
        "fraction-v1", "5/14",
    ),
    BenchmarkDefinition(
        "math-07", "math",
        _answer_prompt("What is the remainder when 7^100 is divided by 13?"),
        "integer-v1", 9,
    ),
    BenchmarkDefinition(
        "math-08", "math",
        _answer_prompt(
            "An arithmetic sequence begins at 11 and has common difference "
            "7. What is its 25th term?"
        ),
        "integer-v1", 179,
    ),
    BenchmarkDefinition(
        "math-09", "math",
        _answer_prompt(
            "A rectangle has perimeter 70 and its length is 5 greater than "
            "its width. What is its area?"
        ),
        "integer-v1", 300,
    ),
    BenchmarkDefinition(
        "math-10", "math",
        _answer_prompt(
            "Six identical machines make 90 parts in 15 minutes. At the "
            "same constant rate, how many minutes do 10 machines need to "
            "make 120 parts?"
        ),
        "integer-v1", 12,
    ),
    BenchmarkDefinition(
        "math-11", "math",
        _answer_prompt(
            "Find the smallest positive integer that leaves remainder 2 "
            "when divided by 5 and remainder 3 when divided by 7."
        ),
        "integer-v1", 17,
    ),
    BenchmarkDefinition(
        "math-12", "math",
        _answer_prompt(
            "Three equally weighted test scores are 78, 84, and 91. A final "
            "exam counts twice as much as each test. What final-exam score "
            "produces a weighted average of 85?"
        ),
        "integer-v1", 86,
    ),
    # Expert science: concept questions with one unambiguous choice.
    BenchmarkDefinition(
        "expert-science-01", "expert-science",
        _answer_prompt(
            "For canonical position x and momentum p, which commutator is "
            "correct? A) [x,p]=0 B) [x,p]=iℏ C) [x,p]=−ℏ D) [x,p]=i"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "expert-science-02", "expert-science",
        _answer_prompt(
            "In Michaelis–Menten kinetics, a purely competitive inhibitor "
            "typically causes which change? A) lower Km and lower Vmax "
            "B) higher apparent Km and unchanged Vmax C) unchanged Km and "
            "lower Vmax D) higher Km and higher Vmax"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "expert-science-03", "expert-science",
        _answer_prompt(
            "A recessive phenotype has frequency 0.04 in a Hardy–Weinberg "
            "population. What is the expected heterozygote frequency? "
            "A) 0.16 B) 0.20 C) 0.32 D) 0.64"
        ),
        "choice-v1", "C",
    ),
    BenchmarkDefinition(
        "expert-science-04", "expert-science",
        _answer_prompt(
            "At 298 K, a tenfold concentration ratio for a monovalent ion "
            "corresponds to approximately what Nernst potential magnitude? "
            "A) 5.9 mV B) 59 mV C) 118 mV D) 590 mV"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "expert-science-05", "expert-science",
        _answer_prompt(
            "Peptides generated from endogenous cytosolic proteins are "
            "normally presented by which pair? A) MHC I to CD8 T cells "
            "B) MHC II to CD8 T cells C) MHC I to CD4 T cells "
            "D) MHC II to B cells"
        ),
        "choice-v1", "A",
    ),
    BenchmarkDefinition(
        "expert-science-06", "expert-science",
        _answer_prompt(
            "For an ideal unimolecular SN1 solvolysis, the observed rate is "
            "primarily proportional to: A) nucleophile concentration only "
            "B) substrate × nucleophile C) substrate concentration only "
            "D) solvent concentration squared"
        ),
        "choice-v1", "C",
    ),
    BenchmarkDefinition(
        "expert-science-07", "expert-science",
        _answer_prompt(
            "Relative to a distant observer, a clock stationary deeper in a "
            "static gravitational potential well runs: A) faster B) slower "
            "C) at the same rate D) backward"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "expert-science-08", "expert-science",
        _answer_prompt(
            "Which is the correct frequentist interpretation of p=0.03? "
            "A) The null has 3% probability B) The alternative has 97% "
            "probability C) 3% of results are false D) If the null were "
            "true, data at least this extreme would occur with probability 3%"
        ),
        "choice-v1", "D",
    ),
    BenchmarkDefinition(
        "expert-science-09", "expert-science",
        _answer_prompt(
            "Wild-type SpCas9 guided to a target adjacent to NGG normally "
            "cuts double-stranded DNA approximately: A) 3 bp upstream of PAM "
            "B) inside the PAM C) 20 bp downstream of PAM D) at the guide's "
            "5′ end only"
        ),
        "choice-v1", "A",
    ),
    BenchmarkDefinition(
        "expert-science-10", "expert-science",
        _answer_prompt(
            "Assume main-sequence luminosity scales as M^3.5 and available "
            "fuel as M. If the Sun's lifetime is 10 Gyr, the approximate "
            "lifetime of a 2-solar-mass star is: A) 0.18 Gyr B) 1.8 Gyr "
            "C) 5 Gyr D) 28 Gyr"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "expert-science-11", "expert-science",
        _answer_prompt(
            "The Reynolds number primarily compares: A) pressure and surface "
            "tension B) compressibility and gravity C) inertial and viscous "
            "effects D) conductive and radiative heat flow"
        ),
        "choice-v1", "C",
    ),
    BenchmarkDefinition(
        "expert-science-12", "expert-science",
        _answer_prompt(
            "In radiometric dating, a mineral's closure temperature marks "
            "the point below which: A) relevant isotopes no longer diffuse "
            "significantly B) the parent isotope begins to exist C) all decay "
            "stops D) daughter isotopes become massless"
        ),
        "choice-v1", "A",
    ),
    # Code tracing: Python 3, one deterministic printed result.
    BenchmarkDefinition(
        "code-tracing-01", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "x = [1, 2]\ny = x\nx += [3]\nprint(len(y), y[-1])"
        ),
        "exact-text-v1", "3 3",
    ),
    BenchmarkDefinition(
        "code-tracing-02", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "funcs = [lambda i=i: i * i for i in range(4)]\n"
            "print(sum(f() for f in funcs))"
        ),
        "exact-text-v1", "14",
    ),
    BenchmarkDefinition(
        "code-tracing-03", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "def f(x, acc=[]):\n    acc.append(x)\n    return sum(acc)\n"
            "print(f(2), f(3))"
        ),
        "exact-text-v1", "2 5",
    ),
    BenchmarkDefinition(
        "code-tracing-04", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "a = [0, 1, 2, 3, 4, 5]\nprint(a[-2:1:-2])"
        ),
        "exact-text-v1", "[4, 2]",
    ),
    BenchmarkDefinition(
        "code-tracing-05", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "d = {i % 3: i for i in range(7)}\nprint(sum(d.values()))"
        ),
        "exact-text-v1", "15",
    ),
    BenchmarkDefinition(
        "code-tracing-06", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "def f(n):\n    return 1 if n <= 1 else n * f(n - 2)\n"
            "print(f(6))"
        ),
        "exact-text-v1", "48",
    ),
    BenchmarkDefinition(
        "code-tracing-07", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "def f():\n    try:\n        return 3\n    finally:\n"
            "        return 5\nprint(f())"
        ),
        "exact-text-v1", "5",
    ),
    BenchmarkDefinition(
        "code-tracing-08", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "g = (i * i for i in range(5) if i % 2)\n"
            "print(next(g) + next(g))"
        ),
        "exact-text-v1", "10",
    ),
    BenchmarkDefinition(
        "code-tracing-09", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "calls = []\ndef mark(x):\n    calls.append(x)\n    return x\n"
            "print(mark(0) and mark(1), calls)"
        ),
        "exact-text-v1", "0 [0]",
    ),
    BenchmarkDefinition(
        "code-tracing-10", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "s = 'abcdef'\nprint(s[1::2][::-1])"
        ),
        "exact-text-v1", "fdb",
    ),
    BenchmarkDefinition(
        "code-tracing-11", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "x = 10\ndef outer():\n    x = 3\n    def inner(y=x):\n"
            "        return y + x\n    x = 4\n    return inner()\n"
            "print(outer())"
        ),
        "exact-text-v1", "7",
    ),
    BenchmarkDefinition(
        "code-tracing-12", "code-tracing",
        _exact_prompt(
            "Under Python 3, what does this print?\n"
            "pairs = zip([1, 2, 3], [4, 5, 6])\n"
            "print(sum(i * a * b for i, (a, b) in "
            "enumerate(pairs, start=1)))"
        ),
        "exact-text-v1", "78",
    ),
    # Instruction following: exact transformations, not open-ended writing.
    BenchmarkDefinition(
        "instruction-following-01", "instruction-following",
        _exact_prompt(
            "Sort `pear fig banana kiwi` by increasing word length, breaking "
            "ties alphabetically, and join the words with `|`."
        ),
        "exact-text-v1", "fig|kiwi|pear|banana",
    ),
    BenchmarkDefinition(
        "instruction-following-02", "instruction-following",
        _exact_prompt(
            "Partition the integers 1 through 6 into a JSON object with keys "
            "`even` and `odd`, each mapped to an ascending array. Return only "
            "the JSON object."
        ),
        "json-v1", {"even": [2, 4, 6], "odd": [1, 3, 5]},
    ),
    BenchmarkDefinition(
        "instruction-following-03", "instruction-following",
        _exact_prompt(
            "Shift every letter in `NETWORK` forward by one position in the "
            "English alphabet, without wrapping or changing case."
        ),
        "exact-text-v1", "OFUXPSL",
    ),
    BenchmarkDefinition(
        "instruction-following-04", "instruction-following",
        _exact_prompt(
            "Take the third character of each word in `planet candle bridge` "
            "and concatenate them in the original order."
        ),
        "exact-text-v1", "ani",
    ),
    BenchmarkDefinition(
        "instruction-following-05", "instruction-following",
        _exact_prompt(
            "Reverse the word order in `red blue green`, uppercase every word, "
            "and separate the result with single spaces."
        ),
        "exact-text-v1", "GREEN BLUE RED",
    ),
    BenchmarkDefinition(
        "instruction-following-06", "instruction-following",
        _exact_prompt(
            "Replace every occurrence of a, e, i, o, or u in `Capability` "
            "with `*`, preserving all other characters and their case."
        ),
        "exact-text-v1", "C*p*b*l*ty",
    ),
    BenchmarkDefinition(
        "instruction-following-07", "instruction-following",
        _exact_prompt(
            "Output exactly three lines. Line 1 is `alpha`, line 2 is `beta`, "
            "and line 3 is `gamma`. Do not use bullets or numbering."
        ),
        "exact-text-v1", "alpha\nbeta\ngamma",
    ),
    BenchmarkDefinition(
        "instruction-following-08", "instruction-following",
        _exact_prompt(
            "Return only a JSON object mapping each name to its character "
            "count: Ada, Grace, Linus."
        ),
        "json-v1", {"Ada": 3, "Grace": 5, "Linus": 5},
    ),
    BenchmarkDefinition(
        "instruction-following-09", "instruction-following",
        _exact_prompt(
            "Rewrite `experiment` with alternating uppercase and lowercase "
            "letters, beginning with uppercase."
        ),
        "exact-text-v1", "ExPeRiMeNt",
    ),
    BenchmarkDefinition(
        "instruction-following-10", "instruction-following",
        _exact_prompt(
            "From `Mississippi`, remove every repeated letter after its first "
            "appearance, comparing letters case-insensitively while preserving "
            "the original case of retained letters."
        ),
        "exact-text-v1", "Misp",
    ),
    BenchmarkDefinition(
        "instruction-following-11", "instruction-following",
        _exact_prompt(
            "Reverse the digits of 4729, add 17 to the resulting integer, and "
            "enclose the final number in square brackets."
        ),
        "exact-text-v1", "[9291]",
    ),
    BenchmarkDefinition(
        "instruction-following-12", "instruction-following",
        _exact_prompt(
            "For `delta alpha charlie bravo`, move the final word to the "
            "front, sort the remaining three words alphabetically, and join "
            "all four with commas and no spaces."
        ),
        "exact-text-v1", "bravo,alpha,charlie,delta",
    ),
    # Multi-step reasoning: original problems requiring chained deductions.
    BenchmarkDefinition(
        "multi-step-reasoning-01", "multi-step-reasoning",
        _answer_prompt(
            "Four people A, B, C, and D occupy positions 1–4 from left to "
            "right. A is left of B. C is not at an end. D is immediately "
            "right of C. B is not adjacent to C. What position contains B?"
        ),
        "integer-v1", 4,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-02", "multi-step-reasoning",
        _answer_prompt(
            "Three boxes are labeled Apples, Oranges, and Mixed, but every "
            "label is wrong. One fruit drawn from the box labeled Mixed is an "
            "apple. What should that box actually be labeled? A) Apples "
            "B) Oranges C) Mixed D) Impossible to know"
        ),
        "choice-v1", "A",
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-03", "multi-step-reasoning",
        _answer_prompt(
            "A knight always tells the truth and a knave always lies. A says "
            "'B is a knave.' B says 'We are different types.' Which is true? "
            "A) both knights B) A knave, B knight C) A knight, B knave "
            "D) both knaves"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-04", "multi-step-reasoning",
        _answer_prompt(
            "Two cities are 300 miles apart. A train leaves the first at "
            "9:00 AM traveling 60 mph. A train leaves the second at 10:00 AM "
            "toward the first at 90 mph. At what time do they meet? Give the "
            "time as HH:MM AM/PM."
        ),
        "exact-answer-v1", "11:36 AM",
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-05", "multi-step-reasoning",
        _answer_prompt(
            "Tasks P, Q, R, S, T must be ordered. P is before Q; R is after "
            "Q; S is before R; Q is before T. Which order is valid? "
            "A) Q P S R T B) S P Q T R C) P R Q S T D) S Q P T R"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-06", "multi-step-reasoning",
        _answer_prompt(
            "Ana is currently twice Ben's age. Six years ago Ana was three "
            "times Ben's age. How old is Ana now?"
        ),
        "integer-v1", 24,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-07", "multi-step-reasoning",
        _answer_prompt(
            "On a grid, move only right or up from (0,0) to (4,3). How many "
            "shortest paths avoid the point (2,1)?"
        ),
        "integer-v1", 17,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-08", "multi-step-reasoning",
        _answer_prompt(
            "Among 120 students, 70 study math, 55 study physics, and 20 study "
            "neither. How many study both math and physics?"
        ),
        "integer-v1", 25,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-09", "multi-step-reasoning",
        _answer_prompt(
            "All lorins are mavens. No mavens are torps. Some selks are "
            "lorins. Which conclusion must follow? A) No selks are torps "
            "B) Some selks are not torps C) All selks are mavens "
            "D) Some torps are not lorins"
        ),
        "choice-v1", "B",
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-10", "multi-step-reasoning",
        _answer_prompt(
            "A three-digit code has digit sum 13. Its hundreds digit is twice "
            "its units digit, and its tens digit is one more than its units "
            "digit. What is the code?"
        ),
        "integer-v1", 643,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-11", "multi-step-reasoning",
        _answer_prompt(
            "An undirected graph has weighted edges A–B=4, A–C=2, C–B=1, "
            "B–D=5, C–D=8, C–E=10, and D–E=2. What is the shortest-path "
            "cost from A to E?"
        ),
        "integer-v1", 10,
    ),
    BenchmarkDefinition(
        "multi-step-reasoning-12", "multi-step-reasoning",
        _answer_prompt(
            "Three fair coins are tossed. Given that at least two are heads, "
            "what is the probability that all three are heads? Give the "
            "fraction in lowest terms."
        ),
        "fraction-v1", "1/4",
    ),
)


def _canonical_config(config: GraderConfig) -> str:
    return json.dumps(
        dataclasses.asdict(config), separators=(",", ":"), sort_keys=True
    )


def grader_config_hash(config: GraderConfig) -> str:
    """Return the SHA-256 identity embedded in the provider-safe bank."""
    return hashlib.sha256(_canonical_config(config).encode("utf-8")).hexdigest()


def build_bank_and_configs(
    definitions: Sequence[BenchmarkDefinition] = DEFINITIONS,
) -> Tuple[Tuple[rnmoncowar.WarmupItem, ...], Tuple[GraderConfig, ...]]:
    """Split prompts from private answers and bind them by hash."""
    bank = []
    configs = []
    for definition in definitions:
        config = GraderConfig(
            definition.item_id, definition.grader_id, definition.expected
        )
        configs.append(config)
        bank.append(
            rnmoncowar.WarmupItem(
                item_id=definition.item_id,
                family=definition.family,
                prompt=definition.prompt,
                grader_id=definition.grader_id,
                grader_config_hash=grader_config_hash(config),
            )
        )
    return tuple(bank), tuple(configs)


def render_bank_jsonl(items: Sequence[rnmoncowar.WarmupItem]) -> str:
    return "".join(
        json.dumps(dataclasses.asdict(item), sort_keys=True) + "\n"
        for item in items
    )


def render_grader_configs_jsonl(configs: Sequence[GraderConfig]) -> str:
    return "".join(_canonical_config(config) + "\n" for config in configs)


def load_grader_configs(path: str) -> Mapping[str, GraderConfig]:
    """Load the private answer file and require unique item identifiers."""
    configs: Dict[str, GraderConfig] = {}
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            try:
                config = GraderConfig(**json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "invalid grader config row %s: %s" % (line_number, error)
                ) from error
            if config.item_id in configs:
                raise ValueError("grader config item ids must be unique")
            configs[config.item_id] = config
    return configs


def validate_grader_configs(
    items: Sequence[rnmoncowar.WarmupItem],
    configs: Mapping[str, GraderConfig],
) -> None:
    """Require every bank hash to match exactly one private answer."""
    if {item.item_id for item in items} != set(configs):
        raise ValueError("grader config ids do not match the warm-up bank")
    for item in items:
        config = configs[item.item_id]
        if config.grader_id != item.grader_id:
            raise ValueError("grader identity mismatch for %s" % item.item_id)
        if grader_config_hash(config) != item.grader_config_hash:
            raise ValueError("grader config hash mismatch for %s" % item.item_id)


def _strip_answer_prefix(response: str) -> str:
    text = response.strip()
    match = re.fullmatch(r"(?is)ANSWER\s*:\s*(.*?)\s*", text)
    return match.group(1).strip() if match else text


def grade_response(grader_id: str, expected: object, response: str) -> bool:
    """Apply one strict, deterministic normalizer."""
    actual = _strip_answer_prefix(response)
    if grader_id == "integer-v1":
        cleaned = actual.replace(",", "")
        return re.fullmatch(r"[-+]?\d+", cleaned) is not None and int(
            cleaned
        ) == int(expected)
    if grader_id == "fraction-v1":
        return re.fullmatch(r"-?\d+/-?\d+", actual) is not None and actual == expected
    if grader_id == "choice-v1":
        return actual.upper() == str(expected).upper()
    if grader_id in ("exact-text-v1", "exact-answer-v1"):
        return actual == expected
    if grader_id == "json-v1":
        try:
            return json.loads(actual) == expected
        except json.JSONDecodeError:
            return False
    raise ValueError("unknown grader %r" % grader_id)


def _last_marked_answer(response: str) -> Optional[str]:
    matches = re.findall(
        r"(?im)^\s*ANSWER\s*:\s*(.*?)\s*$", response.strip()
    )
    return matches[-1].strip() if matches else None


def grade_substantive_answer(
    grader_id: str, expected: object, response: str
) -> bool:
    """
    Audit substantive answers separately from strict format compliance.

    This post-collection diagnostic is intentionally not a replacement for
    the frozen strict grader. It accepts a final ANSWER marker and, for a
    multiple-choice response, a leading option letter followed by option text.
    """
    marked = _last_marked_answer(response)
    actual = marked if marked is not None else response.strip()
    if grader_id == "choice-v1":
        match = re.match(r"^([A-Da-d])(?:\b|\))", actual)
        return match is not None and match.group(1).upper() == str(
            expected
        ).upper()
    if grader_id in (
        "integer-v1",
        "fraction-v1",
        "exact-answer-v1",
    ):
        return grade_response(grader_id, expected, actual)
    return grade_response(grader_id, expected, response)


def grade_observation(
    observation: Mapping[str, object],
    configs: Mapping[str, GraderConfig],
) -> Grade:
    """Grade one saved observation without mutating the raw JSONL row."""
    metadata = dict(observation["metadata"])
    item_id = metadata["item_id"]
    config = configs[item_id]
    provider_status = str(observation["provider_status"])
    response = observation.get("response")
    if provider_status != "ok" or not isinstance(response, str):
        correct = None
        substantive_correct = None
        reason = "PROVIDER_STATUS_%s" % provider_status.upper()
        substantive_reason = reason
    else:
        correct = grade_response(config.grader_id, config.expected, response)
        if metadata["family"] in (
            "math",
            "expert-science",
            "multi-step-reasoning",
        ):
            substantive_correct = grade_substantive_answer(
                config.grader_id, config.expected, response
            )
        else:
            substantive_correct = correct
        reason = "MATCH" if correct else "MISMATCH"
        substantive_reason = (
            "MATCH" if substantive_correct else "MISMATCH"
        )
    return Grade(
        call_id=str(observation["call_id"]),
        item_id=item_id,
        family=metadata["family"],
        route_name=str(observation["route_name"]),
        provider_status=provider_status,
        correct=correct,
        substantive_correct=substantive_correct,
        grader_id=config.grader_id,
        reason=reason,
        substantive_reason=substantive_reason,
    )


def summarize_grades(grades: Sequence[Grade]) -> Mapping[str, object]:
    """Aggregate accuracy by route and route/family."""
    routes: Dict[str, Dict[str, int]] = {}
    route_families: Dict[str, Dict[str, int]] = {}
    for grade in grades:
        route = routes.setdefault(
            grade.route_name,
            {
                "correct": 0,
                "substantive_correct": 0,
                "graded": 0,
                "ungraded": 0,
            },
        )
        key = "%s|%s" % (grade.route_name, grade.family)
        family = route_families.setdefault(
            key,
            {
                "correct": 0,
                "substantive_correct": 0,
                "graded": 0,
                "ungraded": 0,
            },
        )
        for bucket in (route, family):
            if grade.correct is None:
                bucket["ungraded"] += 1
            else:
                bucket["graded"] += 1
                bucket["correct"] += int(grade.correct)
                bucket["substantive_correct"] += int(
                    grade.substantive_correct
                )
    route_rows = []
    for route_name, counts in sorted(routes.items()):
        route_rows.append(
            {
                "route_name": route_name,
                **counts,
                "accuracy": (
                    counts["correct"] / counts["graded"]
                    if counts["graded"]
                    else None
                ),
                "substantive_accuracy": (
                    counts["substantive_correct"] / counts["graded"]
                    if counts["graded"]
                    else None
                ),
            }
        )
    family_rows = []
    for key, counts in sorted(route_families.items()):
        route_name, family = key.split("|", 1)
        family_rows.append(
            {
                "route_name": route_name,
                "family": family,
                **counts,
                "accuracy": (
                    counts["correct"] / counts["graded"]
                    if counts["graded"]
                    else None
                ),
                "substantive_accuracy": (
                    counts["substantive_correct"] / counts["graded"]
                    if counts["graded"]
                    else None
                ),
            }
        )
    return {
        "schema_version": "noesis-warmup-grades-v2",
        "total_observations": len(grades),
        "routes": route_rows,
        "route_families": family_rows,
    }
