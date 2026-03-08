from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

from models import OptimizationContext
from settings import GepaSettings


@dataclass
class GepaCandidate:
    name: str
    content: str
    rationale: str


class GEPAOptimizer:
    """GEPA-backed optimizer that proposes skill edits from failure signals."""

    def __init__(self, settings: GepaSettings, strict_mode: bool = False) -> None:
        self.settings = settings
        self.strict_mode = strict_mode

    def propose_candidates(
        self,
        skill_text: str,
        context: OptimizationContext,
    ) -> List[GepaCandidate]:
        dataset = self._build_dataset(context)
        config = GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=self.settings.max_metric_calls,
                capture_stdio=False,
                display_progress_bar=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=self.settings.reflection_model,
                reflection_minibatch_size=1,
            ),
        )

        def evaluator(candidate: str, example: Dict | None = None) -> Tuple[float, Dict]:
            return self._score_candidate(candidate, example or {})

        try:
            result = optimize_anything(
                seed_candidate=skill_text,
                evaluator=evaluator,
                dataset=dataset,
                objective=self.settings.objective,
                config=config,
            )
            content = self._extract_candidate_text(result.best_candidate)
            rationale = (
                "GEPA reflection using failure-tag and outcome heuristics "
                f"(max_metric_calls={self.settings.max_metric_calls})."
            )
            return [
                GepaCandidate(
                    name=f"{context.skill_version.skill_name}-gepa",
                    content=content,
                    rationale=rationale,
                )
            ]
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError(f"GEPA optimization failed in strict mode: {exc}") from exc
            # Fall back to returning the original skill with diagnostic footer.
            fallback = skill_text.rstrip() + "\n\n<!-- GEPA failed: " + str(exc) + " -->"
            return [
                GepaCandidate(
                    name=f"{context.skill_version.skill_name}-gepa",
                    content=fallback,
                    rationale=f"GEPA optimization failed: {exc}",
                )
            ]

    # ------------------------------------------------------------------
    def _build_dataset(self, context: OptimizationContext) -> List[Dict]:
        dataset: List[Dict] = []
        if context.failure_taxonomy:
            for tag, count in context.failure_taxonomy.items():
                dataset.append(
                    {
                        "failure_tag": tag,
                        "count": count,
                        "prompt": f"Reduce future occurrences of '{tag}' ({count} hits).",
                    }
                )
        else:
            dataset.append(
                {
                    "failure_tag": "general",
                    "count": 1,
                    "prompt": "Improve task planning discipline and output reliability.",
                }
            )
        return dataset

    def _score_candidate(self, candidate_text: str, example: Dict) -> Tuple[float, Dict]:
        text_lower = candidate_text.lower()
        score = 0.35
        notes: List[str] = []
        for keyword in ("plan", "verify", "checklist", "fallback", "examples", "constraints"):
            if keyword in text_lower:
                score += 0.05
        for anti_keyword in ("harbor", "trulens", "promotion gate", "gpa", "gepa"):
            if anti_keyword in text_lower:
                score -= 0.08
                notes.append(f"Contains non-domain framework text: {anti_keyword}")
        tag = example.get("failure_tag")
        if tag and tag.replace("-", " ") in text_lower:
            score += 0.2
            notes.append(f"Addresses {tag}")
        coverage = len(candidate_text.splitlines())
        if coverage > 40:
            score += 0.05
        score = max(0.0, min(score, 1.0))
        return score, {"notes": notes or ["heuristic score"], "example": example}

    def _extract_candidate_text(self, candidate: Dict | str) -> str:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            if "current_candidate" in candidate:
                return candidate["current_candidate"]
            if len(candidate) == 1:
                return next(iter(candidate.values()))
            return "\n".join(str(value) for value in candidate.values())
        return str(candidate)
