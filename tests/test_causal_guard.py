"""Deterministic test suite for evonova-causal-guard."""

from __future__ import annotations

import asyncio

import pytest

from evonova_causal_guard import (
    CausalClaim,
    CausalDAG,
    CycleError,
    DebateArbiter,
    TabuMemory,
    VerdictStatus,
)


def test_cycle_and_self_loop_rejected() -> None:
    dag = CausalDAG([("A", "B"), ("B", "C")])
    with pytest.raises(CycleError):
        dag.add_edge("C", "A")
    with pytest.raises(CycleError):
        dag.add_edge("A", "A")


def test_do_intervention_mutilates_incoming_edges_only() -> None:
    dag = CausalDAG([("Z", "X"), ("Z", "Y"), ("X", "Y")])
    mutilated = dag.do("X")
    assert mutilated.parents("X") == set()
    assert mutilated.children("X") == {"Y"}
    assert mutilated.parents("Y") == {"Z", "X"}
    assert dag.parents("X") == {"Z"}


def test_chain_and_fork_d_separation() -> None:
    chain = CausalDAG([("X", "M"), ("M", "Y")])
    assert chain.d_separated("X", "Y") is False
    assert chain.d_separated("X", "Y", {"M"}) is True

    fork = CausalDAG([("Z", "X"), ("Z", "Y")])
    assert fork.d_separated("X", "Y") is False
    assert fork.d_separated("X", "Y", {"Z"}) is True


def test_collider_d_separation_and_descendant_opening() -> None:
    dag = CausalDAG([("X", "K"), ("Y", "K"), ("K", "D")])
    assert dag.d_separated("X", "Y") is True
    assert dag.d_separated("X", "Y", {"K"}) is False
    assert dag.d_separated("X", "Y", {"D"}) is False


def test_simpsons_paradox_backdoor_adjustment_sets() -> None:
    dag = CausalDAG([("Age", "Drug"), ("Age", "Recovery"), ("Drug", "Recovery")])
    sets = dag.find_backdoor_adjustment_sets("Drug", "Recovery")
    assert sets == [{"Age"}]


def test_m_bias_identifies_empty_set_and_flags_collider() -> None:
    dag = CausalDAG(
        [
            ("U1", "X"),
            ("U1", "M"),
            ("U2", "M"),
            ("U2", "Y"),
            ("X", "Y"),
        ]
    )
    sets = dag.find_backdoor_adjustment_sets("X", "Y")
    assert set() in sets
    biased = dag.detect_collider_bias("X", "Y", {"M"})
    assert biased == ["M"]


def test_detect_collider_bias_on_berkson_descendant() -> None:
    dag = CausalDAG([("Talent", "Admitted"), ("Effort", "Admitted"), ("Admitted", "Badge"), ("Talent", "Effort")])
    assert dag.detect_collider_bias("Talent", "Effort", {"Badge"}) == ["Badge"]


def test_tabu_memory_exact_fuzzy_and_ttl_expiry() -> None:
    clock = [100.0]
    tabu = TabuMemory(similarity_threshold=0.70, ttl_seconds=60.0, time_fn=lambda: clock[0])
    tabu.record_falsified(
        "Ice cream sales directly cause drowning events",
        treatment="IceCream",
        outcome="Drowning",
        reason="Spurious correlation confounded by Temperature",
    )
    assert len(tabu) == 1

    exact = tabu.check("Ice cream sales directly cause drowning events", "IceCream", "Drowning")
    assert exact is not None and exact["match_type"] == "exact"

    fuzzy = tabu.check("Ice cream sales directly cause drowning incidents", "IceCream", "Drowning")
    assert fuzzy is not None and fuzzy["match_type"] == "fuzzy"

    clock[0] = 170.0
    assert tabu.check("Ice cream sales directly cause drowning events", "IceCream", "Drowning") is None
    assert len(tabu) == 0


def test_debate_arbiter_rejects_spurious_claim_and_blocks_repeat_via_tabu() -> None:
    dag = CausalDAG([("Temp", "IceCream"), ("Temp", "Drowning")])
    arbiter = DebateArbiter(dag)

    claim = CausalClaim(
        statement="IceCream causes Drowning",
        treatment="IceCream",
        outcome="Drowning",
    )
    verdict1 = asyncio.run(arbiter.verify(claim))
    assert verdict1.status == VerdictStatus.REJECTED
    assert any("No directed causal path" in v for v in verdict1.violations)
    assert [t["role"] for t in verdict1.trace] == ["Proposer", "Skeptic", "Judge"]

    verdict2 = asyncio.run(arbiter.verify(claim))
    assert verdict2.status == VerdictStatus.REJECTED
    assert verdict2.trace[0]["role"] == "TabuMemory"


def test_debate_arbiter_enforces_backdoor_adjustment_and_accepts_adjusted_claim() -> None:
    dag = CausalDAG([("Severity", "Drug"), ("Severity", "Recovery"), ("Drug", "Recovery")])
    arbiter = DebateArbiter(dag)

    unadjusted = CausalClaim(
        statement="Drug increases Recovery directly without stratification",
        treatment="Drug",
        outcome="Recovery",
        proposed_adjustments=set(),
    )
    v_bad = asyncio.run(arbiter.verify(unadjusted))
    assert v_bad.status == VerdictStatus.REJECTED
    assert {"Severity"} in v_bad.recommended_adjustments

    adjusted = CausalClaim(
        statement="Drug increases Recovery when adjusting for Severity",
        treatment="Drug",
        outcome="Recovery",
        proposed_adjustments={"Severity"},
    )
    v_ok = asyncio.run(arbiter.verify(adjusted))
    assert v_ok.status == VerdictStatus.ACCEPTED
    assert v_ok.violations == []


def test_debate_arbiter_rejects_post_treatment_descendant_adjustment() -> None:
    dag = CausalDAG([("Treatment", "Mediator"), ("Mediator", "Outcome")])
    arbiter = DebateArbiter(dag)
    claim = CausalClaim(
        statement="Adjusting on intermediate Mediator",
        treatment="Treatment",
        outcome="Outcome",
        proposed_adjustments={"Mediator"},
    )
    verdict = asyncio.run(arbiter.verify(claim))
    assert verdict.status == VerdictStatus.REJECTED
    assert any("descendant" in v for v in verdict.violations)


def test_debate_arbiter_supports_async_custom_hooks() -> None:
    dag = CausalDAG([("X", "Y")])

    async def custom_proposer(claim: CausalClaim, ctx: dict) -> str:
        await asyncio.sleep(0)
        return f"Custom thesis for {claim.treatment}->{claim.outcome}"

    arbiter = DebateArbiter(dag, proposer=custom_proposer)
    verdict = asyncio.run(arbiter.verify(CausalClaim(statement="X drives Y", treatment="X", outcome="Y")))
    assert verdict.status == VerdictStatus.ACCEPTED
    assert verdict.trace[0]["message"] == "Custom thesis for X->Y"
