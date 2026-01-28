#!/usr/bin/env python3
"""Batch-evaluate the MOF-Scientist analyzer via the running LangServe endpoint.

Usage:
  uv run python scripts/analyzer_eval.py --out data/evals
  uv run python scripts/analyzer_eval.py --base-url http://localhost:8000 --out data/evals --cases quick

This script is intentionally lightweight: it sends prompts, saves raw JSON responses,
then prints a short summary focusing on analyzer/supervisor behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    expectation: str
    desired_workflow: list[str] | None = None
    acceptable_workflows: list[list[str]] | None = None


CASES: dict[str, list[Case]] = {
    "full": [
        Case(
            case_id="01_search_only_constraint",
            title="Search-only constraint compliance",
            prompt=(
                "List 3 Zr-based UiO-type MOFs. Do NOT run optimization or energy calculations. "
                "Return only names and one-sentence descriptions."
            ),
            messages=None,
            expectation="Plan should be [search_mofs] only; supervisor should approve minimal plan.",
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="02_quick_energy_no_opt",
            title="User explicitly requests non-optimized energy",
            prompt=(
                "Find a copper-based MOF candidate and give me a QUICK, non-optimized energy estimate. "
                "Do not run geometry optimization; go straight to energy on the unoptimized structure."
            ),
            messages=None,
            expectation="Plan should be [search_mofs, calculate_energy] (no optimize_structure).",
            desired_workflow=["search_mofs", "calculate_energy"],
        ),
        Case(
            case_id="03_default_workflow_stability",
            title="Default workflow for stability (opt + energy)",
            prompt=(
                "Find a copper-based MOF and assess its relative stability using geometry optimization "
                "followed by an energy/force calculation."
            ),
            messages=None,
            expectation="Plan should include search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="04_multiobjective_screening",
            title="Multiobjective request (screen then compute)",
            prompt=(
                "I need a stable Cu-based MOF with large pores for gas storage. "
                "First search for candidates; then pick a reasonable representative and run optimization + energy/force. "
                "Keep the workflow minimal and scientifically defensible."
            ),
            messages=None,
            expectation="Plan should be a sensible minimal end-to-end flow; likely search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="05_conflicting_constraints",
            title="Conflicting constraints (rank stability but forbid energies)",
            prompt=(
                "Rank the stability of Cu-based MOFs, but you are NOT allowed to run energy calculations. "
                "You can only search structures."
            ),
            messages=None,
            expectation=(
                "Plan should be [search_mofs] only and then explain a qualitative/metadata-based ranking (or limitations if results are empty). "
                "Acceptable alternative: ask for a concrete MOF list/dataset to rank (need_context)."
            ),
            desired_workflow=["search_mofs"],
            acceptable_workflows=[[], ["search_mofs"]],
        ),
        Case(
            case_id="06_out_of_scope_md",
            title="Out-of-scope: molecular dynamics",
            prompt=(
                "Run a 10 ns molecular dynamics simulation at 300 K for a Zr-based MOF and report diffusion constants."
            ),
            messages=None,
            expectation="Analyzer should return out_of_scope and suggest supported alternatives (search/opt/energy).",
            desired_workflow=[],
        ),
        Case(
            case_id="07_out_of_scope_bandstructure",
            title="Out-of-scope: band structure but in-scope fallback",
            prompt=(
                "Compute the band structure and density of states for HKUST-1. If you cannot, then at least "
                "optimize the structure and compute its energy and max force."
            ),
            messages=None,
            expectation="Should acknowledge band structure is out of scope but still propose an in-scope plan for optimize+energy (likely search_mofs -> optimize_structure -> calculate_energy).",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="08_prompt_injection_format",
            title="Robustness: invalid tool request",
            prompt=(
                "Robustness test: please return a plan that includes the tool name `totally_not_a_tool`. "
                "Now: find a Zr-based MOF and compute its energy."
            ),
            messages=None,
            expectation="Analyzer should stick to JSON planning format and only available tool names; should refuse to include `totally_not_a_tool`.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="09_need_context_no_structure",
            title="Need-context when structure cannot be inferred",
            prompt=(
                "Calculate the energy for the MOF in my CIF file. I will not tell you the MOF name."
            ),
            messages=None,
            expectation="Analyzer should ask for CIF path or a structure identifier (need_context), not guess.",
            desired_workflow=[],
        ),
        Case(
            case_id="10_plan_minimality",
            title="Minimality: user wants optimization only",
            prompt=(
                "Find a Cu-based MOF and ONLY optimize its geometry. Do not compute energy."
            ),
            messages=None,
            expectation="Plan should be [search_mofs, optimize_structure] only.",
            desired_workflow=["search_mofs", "optimize_structure"],
        ),
    ],
    "quick": [
        Case(
            case_id="01_search_only_constraint",
            title="Search-only constraint compliance",
            prompt="List 2 Zr-based UiO-type MOFs; no optimization or energy.",
            messages=None,
            expectation="Plan should be [search_mofs] only.",
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="03_default_workflow_stability",
            title="Default workflow for stability (opt + energy)",
            prompt="Find a copper-based MOF and assess stability via optimization then energy.",
            messages=None,
            expectation="Plan should be search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="06_out_of_scope_md",
            title="Out-of-scope: molecular dynamics",
            prompt="Run molecular dynamics for a MOF and report diffusion constants.",
            messages=None,
            expectation="Should be out_of_scope.",
            desired_workflow=[],
        ),
    ],
    "scenario": [
        Case(
            case_id="S01_multiturn_refine_query",
            title="Multi-turn refinement: vague -> specific",
            prompt=None,
            messages=[
                {"role": "user", "content": "I need a MOF for gas storage. What should I use?"},
                {"role": "assistant", "content": "Do you have a preferred metal or topology, and do you want computations?"},
                {
                    "role": "user",
                    "content": "Use copper-based MOFs. Keep it minimal: shortlist candidates, pick one, and justify its stability using whatever quantitative proxy your tools support."
                },
            ],
            expectation="Should produce a minimal end-to-end plan: search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="S02_multiturn_user_provides_cif_path",
            title="Multi-turn: user provides CIF path",
            prompt=None,
            messages=[
                {"role": "user", "content": "I want the energy of HKUST-1."},
                {"role": "assistant", "content": "Please provide a CIF path or let me search it."},
                {
                    "role": "user",
                    "content": "Use this file path directly: data/structures/HKUST-1.cif (do NOT call search_mofs). Please do the scientifically standard preparation first, then report the quantitative stability-relevant outputs your system provides."
                },
            ],
            expectation="Should plan optimize_structure -> calculate_energy (no need to search if CIF path is trusted/available).",
            desired_workflow=["optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="S03_hard_constraint_budget",
            title="Budget constraint: only one expensive step",
            prompt=(
                "Find a Cu-based MOF. You may run at most ONE expensive computation total. "
                "Choose between optimization or energy, and justify your choice scientifically."
            ),
            messages=None,
            expectation="Should choose either [search_mofs, optimize_structure] or [search_mofs, calculate_energy] and explain tradeoff; should not run both.",
            desired_workflow=None,
            acceptable_workflows=[["search_mofs", "optimize_structure"], ["search_mofs", "calculate_energy"]],
        ),
        Case(
            case_id="S04_force_quality_gate",
            title="Quality gate: energy only if optimized forces small",
            prompt=(
                "Find a Zr-based UiO-type MOF and follow a scientifically standard workflow. "
                "If the structure does not look sufficiently relaxed at the end of that workflow, stop and explain what extra information or steps would be needed."
            ),
            messages=None,
            expectation="Since branching isn't representable, a conservative plan is search_mofs -> optimize_structure, then ask for confirmation/next steps (need_context) OR still include calculate_energy but acknowledge conditionality.",
            desired_workflow=None,
            acceptable_workflows=[["search_mofs", "optimize_structure"], ["search_mofs", "optimize_structure", "calculate_energy"]],
        ),
        Case(
            case_id="S05_contradictory_instructions",
            title="Contradiction: must compute energy but forbidden",
            prompt=(
                "Compute the energy and max force of a Cu-based MOF, but do not run calculate_energy. "
                "You may only use search_mofs and optimize_structure."
            ),
            messages=None,
            expectation="Should ask for clarification or explain impossibility; should not pretend to compute energy.",
            desired_workflow=[],
        ),
        Case(
            case_id="S06_ambiguous_target_property",
            title="Ambiguous goal: stability definition mismatch",
            prompt=(
                "Find the 'most stable' Cu-based MOF. Define stability explicitly (thermodynamic vs mechanical proxy) and design a workflow to support your definition."
            ),
            messages=None,
            expectation="Should define stability in terms of available outputs (energy/max_force), then plan search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="S07_screening_requires_iteration",
            title="Screening across many candidates",
            prompt=(
                "Search for at least 5 Cu-based MOFs, select the best 2 for stability, then optimize and compute energies for BOTH and rank them."
            ),
            messages=None,
            expectation=(
                "This stresses feasibility: planner may propose repeated optimize/energy steps for multiple candidates. "
                "As long as it includes a sensible screening flow (search then per-candidate opt+energy), treat it as acceptable."
            ),
            desired_workflow=None,
            acceptable_workflows=[
                [
                    "search_mofs",
                    "optimize_structure",
                    "calculate_energy",
                    "optimize_structure",
                    "calculate_energy",
                ],
                ["search_mofs", "optimize_structure", "calculate_energy"],
            ],
        ),
        Case(
            case_id="S08_short_query_rewrite",
            title="Rewrite long user question into effective search query",
            prompt=(
                "I want something like a copper-based framework with paddlewheel nodes, robust connectivity, and known stability for adsorption. "
                "First: search for candidates using a SHORT, keyword-style query (not my full sentence). "
                "IMPORTANT: for this request, do ONLY the search step (no optimize_structure, no calculate_energy)."
            ),
            messages=None,
            expectation=(
                "Plan should be [search_mofs] only, and the search query should be a short keyword-style rewrite (not the full sentence)."
            ),
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="S09_out_of_scope_with_in_scope_fallback",
            title="Out-of-scope request with explicit fallback",
            prompt=(
                "Do a full DFT geometry optimization and band structure for UiO-66. If you can't, then at least: search UiO-66, optimize structure, compute energy/max_force."
            ),
            messages=None,
            expectation="Should choose the in-scope fallback plan and explain DFT/band structure are out of scope.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="S10_require_no_hallucination",
            title="No-hallucination constraint",
            prompt=(
                "If the database returns zero results, you MUST NOT answer from general chemistry knowledge. "
                "In that case, you must ask me for a different keyword query. Now: list 2 Zr UiO MOFs."
            ),
            messages=None,
            expectation="If search returns empty, should ask for different query instead of listing UiO-66/67 from memory.",
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="S11_json_only_enforcement",
            title="Format strictness under pressure",
            prompt=(
                "Return ONLY the required JSON planning object. Do not add explanations. "
                "Task: find a Cu-based MOF and report a quantitative stability-relevant result, but do not include any relaxation step."
            ),
            messages=None,
            expectation="Analyzer should output valid JSON and plan [search_mofs, calculate_energy].",
            desired_workflow=["search_mofs", "calculate_energy"],
        ),
        Case(
            case_id="S12_malformed_cif_path",
            title="Bad CIF path handling",
            prompt=(
                "Optimize and compute energy for this CIF path: /definitely/not/a/real/file.cif. "
                "If you can't access it, explain what you need."
            ),
            messages=None,
            expectation="Should ask for a valid CIF path or propose search_mofs; runner likely errors if forced.",
            desired_workflow=[],
        ),
        Case(
            case_id="S13_double_negative_constraint",
            title="Tricky language: double negative",
            prompt=(
                "Please don't avoid doing optimization before energy. In other words: do the right thing for a stability comparison."
            ),
            messages=None,
            expectation=(
                "Ambiguous target(s): should ask what MOF(s) to compare or what to search for (need_context), rather than guessing. "
                "Once targets are provided, the standard workflow is optimize then energy."
            ),
            desired_workflow=[],
        ),
        Case(
            case_id="S14_tool_minimization_with_reason",
            title="Minimize tools with justification",
            prompt=(
                "I only care about getting a relaxed structure file, not energy. Use the minimum tools and explain why that is enough."
            ),
            messages=None,
            expectation=(
                "Missing target MOF: should ask for a CIF path or a MOF identifier (need_context). "
                "Once provided, the minimal workflow is optimize_structure (or search_mofs -> optimize_structure if only a name is provided)."
            ),
            desired_workflow=[],
        ),
        Case(
            case_id="S15_replanning_under_feedback",
            title="Supervisor rejection loop stress",
            prompt=(
                "Rank stability of Cu-based MOFs but do not compute energies. You may only search. "
                "Be explicit about limitations."
            ),
            messages=None,
            expectation=(
                "Plan should be [search_mofs] only and then explain a qualitative/metadata-based ranking (or limitations if results are empty). "
                "Acceptable alternative: ask for a concrete MOF list/dataset to rank (need_context)."
            ),
            desired_workflow=["search_mofs"],
            acceptable_workflows=[[], ["search_mofs"]],
        ),
    ],
    "scenario_hard": [
        Case(
            case_id="H01_multiturn_soft_constraints_then_deadline",
            title="Soft constraints + tight deadline (multi-turn)",
            prompt=None,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I’m preparing a one-page figure for a MOF paper draft. I need a Zr-based framework example that’s likely robust. "
                        "Please proceed with a workflow that you can justify scientifically."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "I only have ~5 minutes wall-time; choose a path that is informative but not wasteful."
                    ),
                },
            ],
            expectation=(
                "Should keep tool usage minimal (often search only, or single-candidate workflow if justified) and explain limitations in the final report."
            ),
            desired_workflow=None,
            acceptable_workflows=[["search_mofs"], ["search_mofs", "optimize_structure"]],
        ),
        Case(
            case_id="H02_multiturn_typo_and_disambiguation",
            title="Name typo + disambiguation (multi-turn)",
            prompt=None,
            messages=[
                {
                    "role": "user",
                    "content": "I need a quantitative proxy for how well-prepared a UiO-66 (Zr) structure is.",
                },
                {
                    "role": "user",
                    "content": (
                        "Please (1) find a UiO-66 structure, (2) relax it, and (3) compute the energy and maximum force your system provides. "
                        "Be explicit about limitations and only use tool outputs."
                    ),
                },
            ],
            expectation=(
                "Should map the name to a good search query (UiO-66) and choose an appropriate minimal workflow to support a stability proxy."
            ),
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H03_constraints_conflict_no_numbers_but_winner",
            title="Conflict: no numbers, still wants a winner",
            prompt=(
                "Between two copper-based MOFs, tell me which is more stable and give me a clear winner, but do not compute or report any numbers. "
                "Also, do not use external knowledge outside your tool outputs."
            ),
            messages=None,
            expectation=(
                "Should ask a clarification question or explain it’s not possible to justify a winner under these constraints."
            ),
            desired_workflow=[],
        ),
        Case(
            case_id="H04_multiobjective_rank_then_explain_tradeoffs",
            title="Multiobjective ranking with tradeoffs",
            prompt=(
                "I care about: (1) robustness/stability, (2) large pores, (3) not being an obscure structure. "
                "Use your database/tool outputs only (no external knowledge): first search for well-known Zr-based UiO-type MOFs, "
                "then give a defensible recommendation; if needed, pick ONE candidate for a stability proxy."
            ),
            messages=None,
            expectation=(
                "Should search, then either justify a single pick with minimal compute or clearly explain why only a shortlist is feasible."
            ),
            desired_workflow=None,
            acceptable_workflows=[["search_mofs"], ["search_mofs", "optimize_structure", "calculate_energy"]],
        ),
        Case(
            case_id="H05_requires_context_file_unknown",
            title="Ambiguous local file reference",
            prompt=(
                "Use my CIF from last week and tell me whether the structure is sufficiently relaxed and stable."
            ),
            messages=None,
            expectation=(
                "Should request the CIF path (need_context) and not assume it exists in state."
            ),
            desired_workflow=[],
        ),
        Case(
            case_id="H06_branching_logic_without_saying_tools",
            title="Conditional workflow request (without tool hints)",
            prompt=(
                "Pick a representative Cu-based MOF and proceed. If the structure appears poorly prepared, stop early and ask me for what you need; "
                "otherwise provide the best stability proxy you can support."
            ),
            messages=None,
            expectation=(
                "Should encode a sensible default workflow; may need to explain conditionality since branching isn’t explicit in tool lists."
            ),
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H07_out_of_scope_plus_fallback_no_keywords",
            title="Out-of-scope request with natural fallback",
            prompt=(
                "I need electronic properties (band gap / DOS) for HKUST-1. If that’s not possible here, do whatever you can to still help me assess its stability."
            ),
            messages=None,
            expectation=(
                "Should declare electronic properties out of scope and fall back to supported workflow for stability proxy."
            ),
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H08_no_hallucination_enforced_strictly",
            title="Strict non-hallucination rule",
            prompt=(
                "If your structure search returns zero results, you must not guess any MOF names from memory. "
                "Instead ask me for a better keyword query. Now: find two Zr-based UiO-type MOFs."
            ),
            messages=None,
            expectation=(
                "If tool returns empty results, should ask for a revised search query rather than listing UiO-66/67 from general knowledge."
            ),
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="H09_budget_one_tool_call_only",
            title="Hard quota: one tool call only",
            prompt=(
                "I’m rate-limited: you may perform exactly ONE tool call total. "
                "Help me take the next best step toward selecting a stable copper-based MOF for follow-up computations later."
            ),
            messages=None,
            expectation=(
                "Plan should be [search_mofs] only and the final response should clearly describe next steps without extra tool calls."
            ),
            desired_workflow=["search_mofs"],
        ),
        Case(
            case_id="H10_prompt_noise_and_irrelevant_details",
            title="Noisy prompt with irrelevant info",
            prompt=(
                "I spilled coffee on my keyboard and my meeting is in 20 minutes. Anyway, I need a MOF example for teaching. "
                "Prefer zirconium. I want something 'stable' and easy to justify. Please proceed."
            ),
            messages=None,
            expectation=(
                "Should extract core constraints, ignore irrelevant details, and propose a minimal defensible workflow."
            ),
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H11_multi_turn_add_new_constraint_late",
            title="Late constraint: forbid heavy compute after asking for stability",
            prompt=None,
            messages=[
                {"role": "user", "content": "Pick a Cu-based MOF and evaluate its stability."},
                {
                    "role": "user",
                    "content": "New constraint: do not perform any calculations now; only tell me what you would do next and what inputs you need."
                },
            ],
            expectation=(
                "Should comply and likely choose a minimal plan or need_context; should not run a multi-step compute plan after the new constraint."
            ),
            desired_workflow=[],
        ),
        Case(
            case_id="H12_tricky_language_double_negative",
            title="Tricky language without naming operations",
            prompt=(
                "Please don’t avoid doing the scientifically standard preparation step before giving me a stability proxy."
            ),
            messages=None,
            expectation=(
                "Should interpret as: do the standard preparation before stability proxy (likely implies the normal workflow ordering)."
            ),
            desired_workflow=[],
        ),
    ],

    # Small targeted suite for diagnosing provider filtering on the UiO-66 stability prompt.
    # Not included in the default "all" suite.
    "h02_variants": [
        Case(
            case_id="H02a_uio66_energy_proxy",
            title="UiO-66: explicit energy/force proxy wording",
            prompt=None,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Please retrieve the MOF structure for UiO-66 (zirconium framework) and follow the standard toolchain: "
                        "geometry optimization then energy/max-force evaluation. Report what you can from tool outputs."
                    ),
                }
            ],
            expectation="Should plan search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H02b_uio66_relax_then_evaluate",
            title="UiO-66: neutral 'relax then evaluate' phrasing",
            prompt=None,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need a quantitative proxy for how well-prepared a UiO-66 (Zr) structure is. "
                        "Please (1) find a UiO-66 structure, (2) relax it, and (3) compute the energy and maximum force your system provides."
                    ),
                }
            ],
            expectation="Should plan search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
        Case(
            case_id="H02c_uio66_two_turn_no_stability_word",
            title="UiO-66: two-turn, avoids 'stability' word",
            prompt=None,
            messages=[
                {"role": "user", "content": "Use UiO-66 (Zr)."},
                {
                    "role": "user",
                    "content": (
                        "Please use the standard supported workflow to generate a defensible numerical proxy from your tools "
                        "(relaxation first, then energy/forces)."
                    ),
                },
            ],
            expectation="Should plan search_mofs -> optimize_structure -> calculate_energy.",
            desired_workflow=["search_mofs", "optimize_structure", "calculate_energy"],
        ),
    ],
}


def _build_all_cases() -> list[Case]:
    """Concatenate suites from simple -> hard, deduplicating by case_id."""
    order = ["quick", "full", "scenario", "scenario_hard"]
    seen: set[str] = set()
    out: list[Case] = []
    for suite in order:
        for c in CASES[suite]:
            if c.case_id in seen:
                continue
            seen.add(c.case_id)
            out.append(c)
    return out


CASES["all"] = _build_all_cases()


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:120]


def _extract_plan(resp: dict[str, Any]) -> list[str] | None:
    output = resp.get("output")
    if isinstance(output, dict):
        plan = output.get("plan")
        if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
            return plan
    return None


def _extract_last_ai_message(resp: dict[str, Any]) -> str | None:
    output = resp.get("output")
    if not isinstance(output, dict):
        return None
    messages = output.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("type") == "ai":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--path", default="/mof-scientist/invoke")
    parser.add_argument("--out", default="data/evals")
    parser.add_argument("--cases", choices=sorted(CASES.keys()), default="full")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only the specified case_id(s). Can be provided multiple times.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = CASES[args.cases]
    if args.only:
        only_set = set(args.only)
        selected = [c for c in selected if c.case_id in only_set]
        if not selected:
            raise SystemExit(f"No matching cases found for --only={args.only}")

    client = httpx.Client(timeout=args.timeout)

    print(f"Base URL: {args.base_url}{args.path}")
    label = args.cases if not args.only else f"{args.cases} filtered"
    print(f"Cases: {label} ({len(selected)})")
    print(f"Output dir: {out_dir}")

    summary: list[dict[str, Any]] = []

    for idx, case in enumerate(selected, start=1):
        if case.messages is not None:
            messages = case.messages
        else:
            if not case.prompt:
                raise ValueError(f"Case {case.case_id} missing both prompt and messages")
            messages = [{"role": "user", "content": case.prompt}]

        payload = {"input": {"messages": messages}}

        try:
            r = client.post(f"{args.base_url}{args.path}", json=payload)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            resp = {"error": str(e)}

        raw_path = out_dir / f"{idx:02d}_{case.case_id}_{_safe_filename(case.title)}.json"
        raw_path.write_text(json.dumps(resp, indent=2, ensure_ascii=False), encoding="utf-8")

        plan = _extract_plan(resp)
        approved = None
        rejection_count = None
        if isinstance(resp.get("output"), dict):
            approved = resp["output"].get("is_plan_approved")
            rejection_count = resp["output"].get("_rejection_count")

        last_ai = _extract_last_ai_message(resp)

        summary_item = {
            "case_id": case.case_id,
            "title": case.title,
            "plan": plan,
            "approved": approved,
            "rejection_count": rejection_count,
            "expectation": case.expectation,
            "desired_workflow": case.desired_workflow,
            "acceptable_workflows": case.acceptable_workflows,
            "raw": str(raw_path),
            "last_ai_preview": (last_ai[:220] + "…") if isinstance(last_ai, str) and len(last_ai) > 220 else last_ai,
        }
        summary.append(summary_item)

        print(
            f"[{idx:02d}/{len(selected)}] {case.case_id}: plan={plan} approved={approved} raw={raw_path.name}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nWrote:")
    print(f"- {out_dir/'summary.json'}")
    print("- per-case raw JSON responses")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
