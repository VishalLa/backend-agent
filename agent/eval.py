from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import Config
from .llm import ChatModel


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
        max_latency_seconds: float = 5.0,
    ) -> EvalResult:
        text = (content or "").lower()
        keywords = tuple(str(keyword).strip().lower() for keyword in expected_keywords if str(keyword).strip())

        if not keywords:
            raise ValueError("expected_keywords must not be empty")

        hits = tuple(keyword for keyword in keywords if keyword in text)
        missing = tuple(keyword for keyword in keywords if keyword not in text)
        coverage = len(hits) / len(keywords)

        max_latency = max_latency_seconds
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


    def run_case(
        self,
        case: EvalCase,
        response_text: str,
        latency_seconds: float
    ) -> EvalResult:
        return self.score_response(
            response_text,
            case.expected_keywords,
            latency_seconds,
            name=case.name,
            agent_key=case.agent_key,
            max_latency_seconds=case.max_latency_seconds,
        )


    def run(
        self,
        cases: list[EvalCase] | None = None
    ) -> dict[str, Any]:
        """Run the suite against the configured local model.

        Cases intentionally use an unbound model: benchmark prompts must not
        modify the project or pause at a confirmation gate. Each case still
        selects its specialist task model, which is what Phase 8 compares.
        """
        suite = cases or self.build_default_suite()
        results: list[EvalResult] = []

        for case in suite:
            case_config = self.config.model_copy(update={"agent_type": case.agent_key})
            model = ChatModel(case_config).get_llm()
            started = time.monotonic()
            
            try:
                response = model.invoke([
                    SystemMessage(
                        content=(
                            "You are being evaluated as a coding specialist. "
                            "Answer the request directly in plain text. Do not call tools."
                        )
                    ),
                    HumanMessage(content=case.prompt),
                ])
                
                content = getattr(response, "content", str(response))
                if isinstance(content, list):
                    content = " ".join(
                        str(part.get("text", part)) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                
                elapsed = time.monotonic() - started
                results.append(self.score_response(
                    str(content), case.expected_keywords, elapsed,
                    name=case.name, agent_key=case.agent_key,
                    max_latency_seconds=case.max_latency_seconds,
                ))
                
            except Exception as exc:  # retain a result for an unavailable model
                elapsed = time.monotonic() - started
                results.append(EvalResult(
                    name=case.name,
                    agent_key=case.agent_key,
                    passed=False,
                    score=0.0,
                    latency_seconds=elapsed,
                    keyword_hits=(),
                    missing_keywords=case.expected_keywords,
                    details=f"Model invocation failed: {exc}",
                ))

        return {
            "passed": sum(result.passed for result in results),
            "failed": sum(not result.passed for result in results),
            "invocation_failures": sum(
                result.details.startswith("Model invocation failed:") for result in results
            ),
            "avg_latency": (
                sum(result.latency_seconds for result in results) / len(results)
                if results else 0.0
            ),
            "results": [
                {
                    "name": result.name,
                    "agent_key": result.agent_key,
                    "passed": result.passed,
                    "score": result.score,
                    "latency_seconds": result.latency_seconds,
                    "keyword_hits": list(result.keyword_hits),
                    "missing_keywords": list(result.missing_keywords),
                    "details": result.details,
                }
                for result in results
            ],
        }
