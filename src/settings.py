from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


@dataclass
class HarborSettings:
    docker_image: Optional[str]
    command: str = "harbor"
    docker_command: str = "docker"
    workspace_mount: str = "/workspace"
    results_mount: str = "/harbor/artifacts"
    extra_env: Dict[str, str] = field(default_factory=dict)

    @property
    def simulation_mode(self) -> bool:
        return not bool(self.docker_image)


@dataclass
class SkillBenchSettings:
    docker_image: Optional[str]
    command: str = "harbor"
    tasks_path: Optional[Path] = None
    docker_command: str = "docker"
    workspace_mount: str = "/workspace"
    results_mount: str = "/skillbench/artifacts"
    extra_env: Dict[str, str] = field(default_factory=dict)

    @property
    def simulation_mode(self) -> bool:
        return not bool(self.docker_image)


@dataclass
class TruLensSettings:
    openai_api_key: Optional[str]
    judge_model: str = "gpt-4o-mini"
    judge_instructions: str | None = None
    strict: bool = False


@dataclass
class UpskillSettings:
    openai_api_key: Optional[str]
    model: str = "gpt-4o-mini"


@dataclass
class GepaSettings:
    reflection_model: str = "openai/gpt-4o-mini"
    objective: str = (
        "Evolve the provided skill instructions to improve task success while "
        "preserving domain-specific guidance and practical examples."
    )
    max_metric_calls: int = 8


@dataclass
class SkillGymSettings:
    harbor: HarborSettings
    skillbench: SkillBenchSettings
    trulens: TruLensSettings
    upskill: UpskillSettings
    gepa: GepaSettings
    strict_real: bool = False


def _parse_extra_env(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    entries = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_file: Path | None = None) -> SkillGymSettings:
    load_kwargs = {"override": False}
    if env_file is not None:
        load_kwargs["dotenv_path"] = env_file
    load_dotenv(**load_kwargs)

    harbor = HarborSettings(
        docker_image=os.getenv("HARBOR_DOCKER_IMAGE"),
        command=os.getenv("HARBOR_CMD", "harbor"),
        docker_command=os.getenv("HARBOR_DOCKER_CMD", "docker"),
        workspace_mount=os.getenv("HARBOR_WORKSPACE_MOUNT", "/workspace"),
        results_mount=os.getenv("HARBOR_RESULTS_MOUNT", "/harbor/artifacts"),
        extra_env=_parse_extra_env(os.getenv("HARBOR_EXTRA_ENV")),
    )
    skillbench = SkillBenchSettings(
        docker_image=os.getenv("SKILLBENCH_DOCKER_IMAGE"),
        command=os.getenv("SKILLBENCH_CMD", "harbor"),
        tasks_path=(
            Path(os.getenv("SKILLBENCH_TASKS_PATH")).expanduser().resolve()
            if os.getenv("SKILLBENCH_TASKS_PATH")
            else None
        ),
        docker_command=os.getenv("SKILLBENCH_DOCKER_CMD", "docker"),
        workspace_mount=os.getenv("SKILLBENCH_WORKSPACE_MOUNT", "/workspace"),
        results_mount=os.getenv("SKILLBENCH_RESULTS_MOUNT", "/skillbench/artifacts"),
        extra_env=_parse_extra_env(os.getenv("SKILLBENCH_EXTRA_ENV")),
    )
    trulens = TruLensSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        judge_model=os.getenv("TRULENS_JUDGE_MODEL", "gpt-4o-mini"),
        judge_instructions=os.getenv("TRULENS_JUDGE_INSTRUCTIONS"),
        strict=_parse_bool(os.getenv("TRULENS_STRICT"), default=False),
    )
    upskill = UpskillSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("UPSKILL_MODEL", "gpt-4o-mini"),
    )
    gepa = GepaSettings(
        reflection_model=os.getenv("GEPA_REFLECTION_MODEL", "openai/gpt-4o-mini"),
        objective=os.getenv(
            "GEPA_OBJECTIVE",
            (
                "Evolve the provided skill using benchmark traces and failure "
                "signals, while keeping guidance domain-specific."
            ),
        ),
        max_metric_calls=int(os.getenv("GEPA_MAX_METRIC_CALLS", "8")),
    )
    return SkillGymSettings(
        harbor=harbor,
        skillbench=skillbench,
        trulens=trulens,
        upskill=upskill,
        gepa=gepa,
        strict_real=_parse_bool(os.getenv("SKILLGYM_STRICT_REAL"), default=False),
    )
