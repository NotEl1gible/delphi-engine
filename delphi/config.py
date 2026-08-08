"""One settings object. Every knob in the engine is here or it does not exist.

Anything read from the environment is read once, in this file, so that "what was this run
configured with" is answerable from a single dump written next to the results. A parameter
discovered later inside a module is a parameter that will not appear in the run record.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Per 1M tokens, (input, output). Used for the cost column, which is not decoration here:
# the adaptive stopping rule is chosen by comparing what a round buys against what it costs.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5": (1.25, 10.0),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "mock": (0.0, 0.0),
}

# `temperature` is a 400 on Sonnet 5 / Opus 5; the depth knob is output_config.effort, which
# in turn errors on Haiku 4.5. A panel that mixes tiers has to carry per-model capability.
MODEL_CAPS: dict[str, dict] = {
    "claude-opus-5": {"temperature": False, "effort": True},
    "claude-sonnet-5": {"temperature": False, "effort": True},
    "claude-haiku-4-5": {"temperature": False, "effort": False},
    "gpt-5": {"temperature": True, "effort": False},
    "groq/llama-3.3-70b-versatile": {"temperature": True, "effort": False},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DELPHI_", env_file=".env",
                                      extra="ignore", protected_namespaces=())

    # --- panel shape -------------------------------------------------------
    provider: str = "mock"                     # mock | litellm
    panel_model: str = "claude-haiku-4-5"
    n_agents: int = 6
    max_rounds: int = 5
    snapshot_rounds: list[int] = Field(default_factory=lambda: [0, 1, 3, 5])
    # Adaptive stopping: halt once the pooled estimate moves less than this in LOG-ODDS
    # between rounds. A probability-point threshold would stop instantly on confident
    # questions and never stop on uncertain ones. The value is set from the ablation curve.
    stop_movement: float = 0.15
    premortem: bool = True

    # --- decision ----------------------------------------------------------
    calibrator_path: str = "artifacts/calibrator.json"
    abstain_tau: float = 1.5                   # log-odds MAD; set from the coverage curve
    max_wrong_rate: float = 0.15

    # --- infrastructure ----------------------------------------------------
    database_url: str = "sqlite:///delphi.db"
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_eager: bool = True                  # tests and single-machine runs
    mlflow_uri: str = "sqlite:///mlflow.db"    # MLflow 3.x refuses a bare ./mlruns
    otlp_endpoint: str = ""                    # e.g. http://localhost:3000/api/public/otel
    otlp_headers: str = ""
    service_name: str = "delphi-engine"

    # --- keys (never written to disk by this project) ----------------------
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # --- determinism -------------------------------------------------------
    seed: int = 7
    # Mock-provider behaviour, used to VALIDATE the instruments against known truth:
    # if the conformity instrument cannot recover mock_herding, it is not measuring herding.
    mock_separation: float = 0.55
    mock_noise: float = 0.9
    mock_herding: float = 0.35
    mock_anchoring: float = 0.4
    # A per-QUESTION error shared by every member, and the most important knob of the five.
    # Independent errors average away as sqrt(n) and correlated ones do not, so a mock with
    # only independent noise makes a six-member panel look near-perfect: the first version
    # scored a Brier of 0.0000, which exercises none of the gates and is not a panel anybody
    # has ever operated. Real members share a base model, a prompt and a worldview, so their
    # mistakes arrive together -- which is most of why swarms disappoint.
    mock_common_bias: float = 1.15


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
