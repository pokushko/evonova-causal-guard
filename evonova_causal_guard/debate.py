"""Multi-agent causal debate arbiter backed by Pearl DAG & Tabu verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, Awaitable, Callable

from .dag import CausalDAG
from .tabu import TabuMemory


class VerdictStatus(str, Enum):
    """Outcome classification of causal hypothesis verification."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"


@dataclass
class CausalClaim:
    """Structured causal hypothesis submitted by an autonomous agent."""

    statement: str
    treatment: str
    outcome: str
    proposed_adjustments: set[str] = field(default_factory=set)
    evidence: str = ""


@dataclass
class DebateVerdict:
    """Verified adjudication output combining symbolic DAG checks and debate trace."""

    status: VerdictStatus
    confidence: float
    violations: list[str]
    recommended_adjustments: list[set[str]]
    trace: list[dict[str, str]]


DebateHook = Callable[[CausalClaim, dict[str, Any]], str | Awaitable[str]]


async def _invoke_hook(hook: DebateHook, claim: CausalClaim, ctx: dict[str, Any]) -> str:
    res = hook(claim, ctx)
    if inspect.isawaitable(res):
        return str(await res)
    return str(res)


class DebateArbiter:
    """Verify causal claims using Tabu lookup, Pearl do-calculus, and Thesis-Antithesis-Judge."""

    def __init__(
        self,
        dag: CausalDAG,
        tabu: TabuMemory | None = None,
        proposer: DebateHook | None = None,
        skeptic: DebateHook | None = None,
        judge: DebateHook | None = None,
    ) -> None:
        self.dag = dag
        self.tabu = tabu if tabu is not None else TabuMemory()
        self._proposer = proposer or self._default_proposer
        self._skeptic = skeptic or self._default_skeptic
        self._judge = judge or self._default_judge

    @staticmethod
    def _default_proposer(claim: CausalClaim, ctx: dict[str, Any]) -> str:
        adj = sorted(claim.proposed_adjustments) or ["none"]
        return f"Thesis: '{claim.statement}' asserts {claim.treatment} -> {claim.outcome} (adjustments={adj})."

    @staticmethod
    def _default_skeptic(claim: CausalClaim, ctx: dict[str, Any]) -> str:
        violations = ctx["violations"]
        if violations:
            return f"Antithesis: Falsified by {len(violations)} structural issue(s): {'; '.join(violations)}"
        return "Antithesis: No structural confounding, collider bias, or acyclic violations detected in DAG."

    @staticmethod
    def _default_judge(claim: CausalClaim, ctx: dict[str, Any]) -> str:
        return f"Judge: Final status {ctx['status'].value} (confidence={ctx['confidence']:.2f})."

    async def verify(self, claim: CausalClaim) -> DebateVerdict:
        """Execute Tabu check, Pearl DAG verification, and adversarial debate."""
        trace: list[dict[str, str]] = []

        # 1. Tabu memory gate
        tabu_hit = self.tabu.check(claim.statement, claim.treatment, claim.outcome)
        if tabu_hit is not None:
            reason = f"Blocked by TabuMemory ({tabu_hit['match_type']} match): {tabu_hit['reason']}"
            trace.append({"role": "TabuMemory", "message": reason})
            return DebateVerdict(
                status=VerdictStatus.REJECTED,
                confidence=0.99,
                violations=[reason],
                recommended_adjustments=[],
                trace=trace,
            )

        # 2. Deterministic Pearl DAG checks
        violations: list[str] = []
        recommended: list[set[str]] = []
        status = VerdictStatus.ACCEPTED

        nodes = self.dag.nodes
        if claim.treatment not in nodes or claim.outcome not in nodes:
            violations.append(
                f"Treatment '{claim.treatment}' or outcome '{claim.outcome}' does not exist in CausalDAG."
            )
            status = VerdictStatus.REJECTED
        elif not self.dag.has_directed_path(claim.treatment, claim.outcome):
            violations.append(
                f"No directed causal path from '{claim.treatment}' to '{claim.outcome}' (spurious correlation)."
            )
            status = VerdictStatus.REJECTED
        else:
            collider_hits = self.dag.detect_collider_bias(
                claim.treatment, claim.outcome, claim.proposed_adjustments
            )
            if collider_hits:
                violations.append(
                    f"Conditioning on {collider_hits} induces collider bias between '{claim.treatment}' and '{claim.outcome}'."
                )
                status = VerdictStatus.REJECTED

            desc_of_t = self.dag.descendants(claim.treatment)
            bad_descendants = sorted(claim.proposed_adjustments & desc_of_t)
            if bad_descendants:
                violations.append(
                    f"Proposed adjustment set includes post-treatment descendant(s) {bad_descendants}."
                )
                status = VerdictStatus.REJECTED

            g_under = self.dag.without_outgoing(claim.treatment)
            recommended = self.dag.find_backdoor_adjustment_sets(claim.treatment, claim.outcome)
            if not recommended:
                violations.append(
                    f"Causal effect of '{claim.treatment}' on '{claim.outcome}' is not identifiable via backdoor criterion."
                )
                status = VerdictStatus.NEEDS_EVIDENCE
            elif not g_under.d_separated(claim.treatment, claim.outcome, claim.proposed_adjustments):
                violations.append(
                    f"Backdoor paths remain open under proposed adjustments {sorted(claim.proposed_adjustments)}."
                )
                status = VerdictStatus.REJECTED

        confidence = 0.95 if status == VerdictStatus.ACCEPTED else (0.92 if status == VerdictStatus.REJECTED else 0.55)
        ctx: dict[str, Any] = {
            "status": status,
            "confidence": confidence,
            "violations": violations,
            "recommended_adjustments": recommended,
        }

        # 3. Thesis -> Antithesis -> Judge debate loop
        thesis = await _invoke_hook(self._proposer, claim, ctx)
        trace.append({"role": "Proposer", "message": thesis})

        antithesis = await _invoke_hook(self._skeptic, claim, ctx)
        trace.append({"role": "Skeptic", "message": antithesis})

        ruling = await _invoke_hook(self._judge, claim, ctx)
        trace.append({"role": "Judge", "message": ruling})

        if status == VerdictStatus.REJECTED and violations:
            self.tabu.record_falsified(
                claim.statement,
                claim.treatment,
                claim.outcome,
                reason=violations[0],
            )

        return DebateVerdict(
            status=status,
            confidence=confidence,
            violations=violations,
            recommended_adjustments=recommended,
            trace=trace,
        )
