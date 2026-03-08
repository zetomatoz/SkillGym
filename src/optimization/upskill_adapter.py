from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, List, Tuple

from openai import OpenAI

from models import OptimizationContext, SkillVersion


@dataclass
class SkillCandidate:
    name: str
    content: str
    rationale: str


class UpskillOptimizer:
    """Trace-aware skill generator with OpenAI-backed candidate synthesis."""

    def __init__(
        self,
        max_candidates: int = 1,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        strict_mode: bool = False,
    ) -> None:
        self.max_candidates = max_candidates
        self.model = model
        self.strict_mode = strict_mode
        self._client = OpenAI(api_key=api_key) if api_key else None

    def propose_candidates(
        self,
        skill_text: str,
        context: OptimizationContext,
    ) -> List[SkillCandidate]:
        if self._client is not None:
            candidate = self._propose_with_openai(skill_text, context)
            if candidate is not None:
                return [candidate]
            if self.strict_mode:
                raise RuntimeError(
                    "Strict mode enabled and OpenAI-based Upskill generation failed."
                )
        elif self.strict_mode:
            raise RuntimeError(
                "Strict mode requires OPENAI_API_KEY for Upskill generation."
            )
        frontmatter, body = self._split_frontmatter(skill_text)
        body = self._normalize_text(body)
        recommendations = self._build_recommendations(context)
        title = self._extract_title(body) or "Skill"
        intro = self._extract_intro(body)
        when_to_use = self._extract_section(body, "When to use")
        tools = self._extract_section(body, "Tools")
        examples = self._extract_section(body, "Examples")
        notes = self._extract_section(body, "Notes")

        when_to_use_block = when_to_use or "- Use this skill for domain-specific tasks in this area."
        tools_block = tools or "- Add concrete tools/libraries relevant to this domain."
        examples_block = examples or "Add executable examples for the most common task variants."
        notes_block = notes or "- Keep outputs validated and formatted according to user requirements."

        rewritten_body = dedent(
            f"""
            # {title}

            {intro}

            ## When to use
            {when_to_use_block}

            ## Tools
            {tools_block}

            ## Examples
            {examples_block}

            ## Standard Workflow
            1. Restate the user goal and output contract before taking actions.
            2. Create a short plan with explicit checkpoints.
            3. Execute the smallest useful action first, then iterate with verification.
            4. Before final output, run a compact correctness and formatting check.

            ## Diagnostics from Recent Runs
            {recommendations['failures']}

            ## Improvement Priorities
            {recommendations['adjustments']}

            ## Failure Tag Playbook
            {recommendations['playbook']}

            ## Quality Guardrails
            - Keep instructions domain-specific and action-oriented.
            - Avoid framework/meta commentary unless the user asks for it.
            - Prefer concrete examples and explicit checks over generic advice.

            ## Notes
            {notes_block}
            """
        ).strip()

        rewritten_skill = (
            f"{frontmatter}\n\n{rewritten_body}".strip()
            if frontmatter
            else rewritten_body
        )

        rationale = (
            "Derived from Upskill heuristic: fully rewrote the skill to incorporate failure-driven improvements while preserving domain focus."
        )

        return [
                SkillCandidate(
                    name=f"{context.skill_version.skill_name}-upskill",
                    content=rewritten_skill,
                    rationale=rationale,
                )
        ]

    def _propose_with_openai(
        self,
        skill_text: str,
        context: OptimizationContext,
    ) -> SkillCandidate | None:
        assert self._client is not None
        context_payload = self._build_llm_context(context)
        prompt = dedent(
            f"""
            You are optimizing an agent SKILL.md.
            Return ONLY valid JSON with:
            {{
              "candidate_skill": "<full SKILL.md text>",
              "rationale": "<short explanation>"
            }}

            Requirements:
            - Preserve the original intent of the skill.
            - Improve failure modes from trace diagnostics.
            - Keep guidance concise and operational.
            - Include explicit checks for planning, tool-use, and output-format compliance.
            - Do not include markdown code fences around JSON.

            Baseline SKILL.md:
            {skill_text}

            Optimization context (JSON):
            {context_payload}
            """
        ).strip()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=3000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            content = self._extract_text(response)
            parsed = json.loads(content)
            candidate_skill = parsed["candidate_skill"].strip()
            rationale = parsed.get("rationale", "OpenAI-based Upskill generation.")
            if not candidate_skill:
                return None
            return SkillCandidate(
                name=f"{context.skill_version.skill_name}-upskill",
                content=candidate_skill,
                rationale=rationale,
            )
        except Exception:
            return None

    def _build_llm_context(self, context: OptimizationContext) -> str:
        gpa = [
            {
                "aggregate_gpa": score.aggregate_gpa,
                "goal_fulfillment": score.goal_fulfillment,
                "plan_quality": score.plan_quality,
                "plan_adherence": score.plan_adherence,
                "execution_efficiency": score.execution_efficiency,
                "logical_consistency": score.logical_consistency,
                "failure_tags": score.failure_tags,
            }
            for score in context.gpa_breakdown[:10]
        ]
        traces = []
        for trace in context.failing_traces[:5]:
            traces.append(
                {
                    "task_id": trace.task_id,
                    "goal": trace.goal,
                    "final_result": trace.final_result,
                }
            )
        payload = {
            "benchmark_summary": context.benchmark_summary,
            "failure_taxonomy": context.failure_taxonomy,
            "gpa_breakdown": gpa,
            "sample_failing_traces": traces,
        }
        return json.dumps(payload, indent=2)

    def _extract_text(self, response: object) -> str:
        choices = getattr(response, "choices", []) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
        return ""

    # ------------------------------------------------------------------
    def _build_recommendations(self, context: OptimizationContext) -> Dict[str, str]:
        failure_counts = context.failure_taxonomy or {}
        if not failure_counts:
            failure_summary = "No major failures observed; focus on efficiency tuning."
        else:
            ordered = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
            failure_summary = "\n".join(
                f"- {tag}: {count} occurrences" for tag, count in ordered
            )
        adjustments = "\n".join(
            [
                "- Strengthen decision points for tool/strategy selection before execution.",
                "- Add compact verification checks before final output.",
                "- Add fallback behavior when expected data/files are missing or malformed.",
            ]
        )
        playbook = "\n".join(
            [
                "- wrong-tool-selection: Enumerate preferred tool order before acting.",
                "- missing-plan: Fail fast if no plan exists; generate a 4+ step plan.",
                "- formatting-noncompliance: Restate output contract before final answer.",
                "- redundant-actions: Collapse adjacent tool calls unless new info appears.",
            ]
        )
        return {
            "failures": failure_summary,
            "adjustments": adjustments,
            "playbook": playbook,
        }

    def _infer_skill_focus(self, skill_text: str) -> str:
        for line in skill_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return "the current skill domain"

    def _extract_title(self, skill_text: str) -> str:
        for line in skill_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return ""

    def _extract_intro(self, skill_text: str) -> str:
        lines = skill_text.splitlines()
        saw_h1 = False
        intro_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                saw_h1 = True
                continue
            if not saw_h1:
                continue
            if stripped.startswith("## "):
                break
            intro_lines.append(line)
        intro = "\n".join(intro_lines).strip()
        if intro:
            return intro
        title = self._extract_title(skill_text) or "this skill"
        return f"This skill provides focused, practical guidance for {title.lower()} tasks."

    def _split_frontmatter(self, skill_text: str) -> Tuple[str, str]:
        lines = skill_text.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return "", skill_text.strip()
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter = "\n".join(lines[: idx + 1]).strip()
                body = "\n".join(lines[idx + 1 :]).strip()
                return frontmatter, body
        return "", skill_text.strip()

    def _extract_section(self, skill_text: str, section_name: str) -> str:
        lines = skill_text.splitlines()
        target = section_name.strip().lower()
        start_idx = -1
        for idx, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped == f"## {target}":
                start_idx = idx + 1
                break
        if start_idx == -1:
            return ""

        section_lines: List[str] = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith("## "):
                break
            section_lines.append(line)
        return self._normalize_text("\n".join(section_lines).strip())

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = dedent(text).strip()
        cleaned_lines = [line.rstrip() for line in normalized.splitlines()]
        return "\n".join(cleaned_lines).strip()
