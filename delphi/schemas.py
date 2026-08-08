"""Typed contracts for everything that crosses a boundary.

Every agent returns a validated object, never a blob of prose to be parsed. That choice is
load-bearing for the measurements rather than a matter of taste: an agent whose probability is
recovered by a regular expression fails in a way that looks like disagreement, and a panel
whose disagreement metric is partly parser noise cannot be used to decide when to abstain.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["forecast", "abstain"]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    resolution_date: str
    outcome: int | None = None          # 1 / 0 once resolved, None for a live question
    split: Literal["dev", "test", "live"] = "live"
    source: str = "authored"
    base_rate_hint: float | None = None

    @field_validator("outcome")
    @classmethod
    def _binary(cls, v):
        if v not in (None, 0, 1):
            raise ValueError("outcome must be 0, 1 or None")
        return v


class AgentVerdict(BaseModel):
    """What one panel member returns in one round."""

    model_config = ConfigDict(extra="ignore")

    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    key_factor: str = ""
    # The agent's own claim about how sure it is. Recorded and deliberately NOT used by the
    # abstention gate: self-reported confidence has been measured to move the wrong way under
    # degradation in a sibling project, so the gate is driven by panel disagreement instead.
    self_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    persona: str
    model: str
    round: int
    verdict: AgentVerdict
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    error: str | None = None


class RoundSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int
    pooled: float
    spread: float
    movement: float | None = None
    probabilities: list[float] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0


class Forecast(BaseModel):
    """The product's output. `p` is calibrated; `p_raw` is the pool before calibration, kept
    so that a bad forecast can be attributed to the panel or to the calibrator rather than
    argued about."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    decision: Decision
    p: float | None
    p_raw: float
    spread: float
    rounds_used: int
    stopped_early: bool
    snapshots: list[RoundSnapshot] = Field(default_factory=list)
    turns: list[AgentTurn] = Field(default_factory=list)
    calibrator: dict = Field(default_factory=dict)
    evidence_used: bool = False
    anchor: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    trace_id: str = ""
    model: str = ""
    arm: str = "panel"


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    resolution_date: str | None = None
    anchor: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: bool = False
    n_agents: int | None = None
    max_rounds: int | None = None
