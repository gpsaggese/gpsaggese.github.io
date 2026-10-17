#!/usr/bin/env python3
"""
Generate the deterministic Noesis v1 candidate-canary bank.

The generated questions are candidates, not trusted quality evidence. Live
calibration decides which candidates are safe to activate.
"""

import argparse
import json
import os
import random
from typing import Callable, Dict, Mapping, Sequence, Tuple

BANK_VERSION = "v1"
CANARIES_PER_FAMILY = 50


def _record(
    family: str,
    number: int,
    prompt: str,
    answer: str,
    normalizer: str,
    seed: int,
) -> Dict[str, object]:
    """
    Build one JSON-compatible canary record.
    """
    return {
        "id": "%s-%s-%03d" % (family, BANK_VERSION, number),
        "bank_version": BANK_VERSION,
        "family": family,
        "prompt": prompt,
        "answer": answer,
        "normalizer": normalizer,
        "generator_seed": seed,
    }


def _arithmetic_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate a nested integer-arithmetic question.
    """
    generator = random.Random(seed)
    a = generator.randint(17, 97)
    b = generator.randint(7, 31)
    c = generator.randint(101, 999)
    d = generator.randint(8, 20)
    e = generator.randint(1, 6)
    f = generator.randint(10, 200)
    answer = ((a * b + c) * (d - e)) - f
    prompt = (
        "Compute ((%d * %d) + %d) * (%d - %d) - %d. "
        "Return only the final integer." % (a, b, c, d, e, f)
    )
    return _record("arithmetic", number, prompt, str(answer), "integer", seed)


def _modular_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate a multi-step modular-arithmetic question.
    """
    generator = random.Random(seed)
    value = generator.randint(1000, 9999)
    modulus = generator.choice((83, 89, 97, 101, 103))
    steps = []
    for _ in range(4):
        multiplier = generator.randint(7, 29)
        addend = generator.randint(11, 97)
        steps.append((multiplier, addend))
    initial_value = value
    for multiplier, addend in steps:
        value = (value * multiplier + addend) % modulus
    instructions = "; then ".join(
        "x = (x * %d + %d) mod %d" % (multiplier, addend, modulus)
        for multiplier, addend in steps
    )
    prompt = (
        "Start with x = %d. Apply these updates in order: %s. "
        "Return only the final integer." % (initial_value, instructions)
    )
    return _record("modular", number, prompt, str(value), "integer", seed)


def _string_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate an ordered string-transformation question.
    """
    generator = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    source = "".join(generator.choice(alphabet) for _ in range(14))
    rotation = generator.randint(2, 5)
    rotated = source[rotation:] + source[:rotation]
    middle_reversed = rotated[:2] + rotated[2:10][::-1] + rotated[10:]
    positions = (2, 4, 7, 9, 12, 14)
    answer = "".join(middle_reversed[position - 1] for position in positions)
    prompt = (
        "Start with %s. Rotate it left by %d characters. Reverse the characters "
        "in positions 3 through 10 inclusive, using one-based positions. From "
        "that result, concatenate the characters at one-based positions "
        "2, 4, 7, 9, 12, and 14. Return only the resulting string."
        % (source, rotation)
    )
    return _record("string", number, prompt, answer, "alnum_lower", seed)


def _ordering_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate a list-reordering question.
    """
    generator = random.Random(seed)
    tokens = generator.sample(
        [
            "amber",
            "birch",
            "coral",
            "denim",
            "ember",
            "frost",
            "grove",
            "hazel",
            "ivory",
            "jade",
            "khaki",
            "lilac",
        ],
        8,
    )
    source = list(tokens)
    moved_position = generator.randint(3, 7)
    moved = tokens.pop(moved_position - 1)
    tokens.insert(0, moved)
    tokens[2:7] = reversed(tokens[2:7])
    rotation = generator.randint(1, 3)
    tokens = tokens[-rotation:] + tokens[:-rotation]
    prompt = (
        "Start with this ordered list: %s. Move the item at one-based position "
        "%d to the front. Reverse positions 3 through 7 inclusive. Then rotate "
        "the full list right by %d positions. Return the final items in order, "
        "separated by single spaces."
        % (" ".join(source), moved_position, rotation)
    )
    return _record(
        "ordering", number, prompt, " ".join(tokens), "alnum_lower", seed
    )


def _code_trace_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate a short Python-loop tracing question.
    """
    generator = random.Random(seed)
    values = [generator.randint(2, 19) for _ in range(7)]
    modulus = generator.randint(3, 5)
    total = 0
    for index, value in enumerate(values):
        if (index + value) % modulus == 0:
            total += value * (index + 1)
        else:
            total -= value
    prompt = (
        "What does this Python code print? Return only the printed integer.\n\n"
        "values = %r\n"
        "total = 0\n"
        "for index, value in enumerate(values):\n"
        "    if (index + value) %% %d == 0:\n"
        "        total += value * (index + 1)\n"
        "    else:\n"
        "        total -= value\n"
        "print(total)" % (values, modulus)
    )
    return _record("code-trace", number, prompt, str(total), "integer", seed)


def _table_canary(number: int, seed: int) -> Dict[str, object]:
    """
    Generate a table-filtering and aggregation question.
    """
    generator = random.Random(seed)
    rows = []
    for index in range(6):
        rows.append(
            (
                chr(ord("A") + index),
                generator.randint(2, 15),
                generator.randint(3, 18),
                generator.randint(1, 9),
            )
        )
    threshold = generator.randint(8, 13)
    selected = [row for row in rows if row[1] % 2 == 1 and row[2] >= threshold]
    answer = sum(
        x_value * y_value - penalty for _, x_value, y_value, penalty in selected
    )
    rendered_rows = "; ".join("%s=(x:%d,y:%d,p:%d)" % row for row in rows)
    prompt = (
        "Rows are %s. Keep rows where x is odd and y is at least %d. For each "
        "kept row compute x*y-p, then sum those values. An empty selection sums "
        "to 0. Return only the final integer." % (rendered_rows, threshold)
    )
    return _record("table", number, prompt, str(answer), "integer", seed)


_GENERATORS: Mapping[str, Callable[[int, int], Dict[str, object]]] = {
    "arithmetic": _arithmetic_canary,
    "code-trace": _code_trace_canary,
    "modular": _modular_canary,
    "ordering": _ordering_canary,
    "string": _string_canary,
    "table": _table_canary,
}


def build_bank() -> Dict[str, Tuple[Dict[str, object], ...]]:
    """
    Return all bank records grouped by family.
    """
    bank = {}
    for family_index, (family, generator) in enumerate(_GENERATORS.items()):
        records = []
        for number in range(1, CANARIES_PER_FAMILY + 1):
            seed = 100000 * (family_index + 1) + number
            records.append(generator(number, seed))
        bank[family] = tuple(records)
    return bank


def write_bank(
    output_directory: str,
    bank: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """
    Write one stable JSONL file per family.
    """
    os.makedirs(output_directory, exist_ok=True)
    for family, records in bank.items():
        path = os.path.join(output_directory, "%s.jsonl" % family)
        with open(path, "w", encoding="utf-8") as file_:
            for record in records:
                file_.write(json.dumps(record, sort_keys=True) + "\n")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive one JSONL file per family.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse()
    write_bank(arguments.output_dir, build_bank())
