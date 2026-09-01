from __future__ import annotations

from dataclasses import dataclass

from config import Config


@dataclass(frozen=True)
class EvalCase:
    """A single benchmark case used to validate the local agent stack."""

    name: str
    agent_key: str
    prompt: str
    expected_keywords: tuple[str, ...]
    provider: str = "local"
    timeout_seconds: float = 30.0
    max_latency_seconds: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", (self.name or "unnamed").strip())
        object.__setattr__(self, "agent_key", (self.agent_key or "backend").strip().lower())
        object.__setattr__(self, "provider", (self.provider or "local").strip().lower())
        if self.provider != "local":
            raise ValueError("EvalCase provider must be 'local' for this project.")
        if not self.expected_keywords:
            raise ValueError("expected_keywords must not be empty")


@dataclass(frozen=True)
class EvalResult:
    """The result of evaluating one response against a benchmark case."""

    name: str
    agent_key: str
    passed: bool
    score: float
    latency_seconds: float
    keyword_hits: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    details: str = ""


class EvalHarness:
    """Lightweight local benchmark harness for pre-tuning validation.

    This scope intentionally stays small and deterministic: it checks whether the
    response contains the expected domain keywords and whether the latency stays
    under a practical local-LLM budget. That gives a stable baseline before more
    expensive prompt or model tuning work begins.
    """

    def __init__(self, config: Config) -> None:
        self.config = config


    def build_default_suite(self) -> list[EvalCase]:
        cases = [
            EvalCase(
                name="backend-routing",
                agent_key="backend",
                prompt="Inspect the auth route and decide whether the password reset flow is safe.",
                expected_keywords=("auth", "route", "reset"),
            ),
            EvalCase(
                name="ml-routing",
                agent_key="ml",
                prompt="Review the dataset split logic and explain how to evaluate model quality.",
                expected_keywords=("dataset", "evaluate", "model"),
            ),
            EvalCase(
                name="git-routing",
                agent_key="git",
                prompt="Summarize the repo state and tell me which branch changed most recently.",
                expected_keywords=("branch", "repo", "status"),
            ),
            EvalCase(
                name="algorithms-routing",
                agent_key="algorithms",
                prompt="Explain the algorithmic tradeoff between binary search and brute force scanning.",
                expected_keywords=("binary", "search", "algorithm"),
            ),
        ]
        return cases


    def score_response(
        self,
        content: str,
        expected_keywords: tuple[str, ...],
        latency_seconds: float,
        *,
        name: str = "unnamed",
        agent_key: str = "backend",
    ) -> EvalResult:
        text = (content or "").lower()
        keywords = tuple(str(keyword).strip().lower() for keyword in expected_keywords if str(keyword).strip())

        if not keywords:
            raise ValueError("expected_keywords must not be empty")

        hits = tuple(keyword for keyword in keywords if keyword in text)
        missing = tuple(keyword for keyword in keywords if keyword not in text)
        coverage = len(hits) / len(keywords)

        max_latency = 5.0
        latency_score = 1.0 if latency_seconds <= max_latency else max(0.0, 1.0 - (latency_seconds - max_latency) / max_latency)
        score = max(0.0, min(1.0, coverage * 0.8 + latency_score * 0.2))
        passed = coverage == 1.0 and latency_seconds <= max_latency

        return EvalResult(
            name=name,
            agent_key=agent_key,
            passed=passed,
            score=score,
            latency_seconds=float(latency_seconds),
            keyword_hits=hits,
            missing_keywords=missing,
            details=(
                f"Matched {len(hits)}/{len(keywords)} keywords; "
                f"latency={latency_seconds:.2f}s; "
                f"passed={passed}"
            ),
        )


    def run_case(self, case: EvalCase, response_text: str, latency_seconds: float) -> EvalResult:
        return self.score_response(
            response_text,
            case.expected_keywords,
            latency_seconds,
            name=case.name,
            agent_key=case.agent_key,
        )
