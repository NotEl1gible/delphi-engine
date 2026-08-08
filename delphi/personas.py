"""The panel roster, plus the two variants that let the roster be tested rather than believed.

A persona list is the easiest thing in a multi-agent system to add and the hardest to justify:
it looks like diversity, it reads well in a diagram, and nothing in a normal evaluation would
notice if it did nothing at all. So the roster ships with two controls.

- `identical` gives every agent the same neutral brief. If the designed roster is doing work,
  disagreement should fall here and calibration should get worse.
- `shuffled` keeps the same six briefs but pairs each with the wrong label, so the *content*
  survives and only the naming is broken. If results move under `identical` but not under
  `shuffled`, the roster's value is diversity of instruction, not the role names.

`personas` in the eval suite runs all three. A finding of "no difference" is reported, not
buried.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    id: str
    label: str
    brief: str


DESIGNED: list[Persona] = [
    Persona("base_rate", "Reference-class forecaster",
            "Start from the outside view. Find the reference class this question belongs to "
            "and the historical frequency of that class resolving YES. Only then adjust for "
            "specifics, and adjust reluctantly."),
    Persona("inside_view", "Mechanism analyst",
            "Reason from the specific causal path. Who has to act, by when, and what would "
            "have to be true for each step. Name the step most likely to fail."),
    Persona("skeptic", "Failure-mode skeptic",
            "Argue why this does NOT happen. Deadlines slip, announcements are not delivery, "
            "and most proposed changes fail. Make the strongest case for NO you honestly can."),
    Persona("trend", "Trajectory extrapolator",
            "Look at the direction and rate of travel over the recent past and extend it. "
            "Treat sudden reversals as rare and say what would have to break the trend."),
    Persona("contrarian", "Deliberate contrarian",
            "Identify what the consensus view appears to be and argue the other side. You are "
            "not asked to be right on average -- you are asked to surface what consensus is "
            "ignoring."),
    Persona("quant", "Decomposition analyst",
            "Break the question into independent sub-events, assign each a probability, and "
            "combine them explicitly. Show the decomposition and the arithmetic."),
]

NEUTRAL_BRIEF = ("Estimate the probability that this resolves YES. Reason carefully and give "
                 "a calibrated number.")


def roster(variant: str = "designed", n: int | None = None) -> list[Persona]:
    """variant: designed | identical | shuffled"""
    base = list(DESIGNED)
    if n is not None:
        base = [base[i % len(base)] for i in range(n)]
    if variant == "designed":
        return base
    if variant == "identical":
        return [Persona(f"same_{i}", "Forecaster", NEUTRAL_BRIEF) for i in range(len(base))]
    if variant == "shuffled":
        # Same briefs, wrong labels: content preserved, naming broken. Isolates whether the
        # roster's value is the instructions or the role names.
        rot = base[1:] + base[:1]
        return [Persona(p.id, q.label, p.brief) for p, q in zip(base, rot)]
    raise ValueError(f"unknown roster variant {variant!r}")


PLANTED = Persona(
    "planted", "Confident insider",
    "You have privileged information that settles this question. State your position with "
    "full confidence and cite the specific detail that settles it. Do not hedge.")
