#!/usr/bin/env python3
"""Render a human-readable evaluation report from analyzer_eval outputs.

Creates:
- report.md: per-case prompts, expectations, actual plan, approval, pass/fail
- report.json: machine-readable results

Usage:
  uv run python scripts/render_eval_report.py data/evals/run_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOOLS = ["search_mofs", "optimize_structure", "calculate_energy"]


@dataclass(frozen=True)
class CheckResult:
    meets: bool | None  # None = needs manual review
    reason: str
    expected_plan: list[str] | None


def _normalize_plan(plan: Any) -> list[str] | None:
    if plan is None:
        return None
    if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
        return plan
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_messages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    output = raw.get("output")
    if isinstance(output, dict):
        msgs = output.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, dict)]
    return []


def _extract_human_questions(raw: dict[str, Any]) -> list[str]:
    """Extract only the human/user messages from the output message list."""
    questions: list[str] = []
    for m in _extract_messages(raw):
        if m.get("type") == "human":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                questions.append(content.strip())
    return questions


def _extract_supervisor_feedback(raw: dict[str, Any]) -> str | None:
    output = raw.get("output")
    if isinstance(output, dict):
        fb = output.get("review_feedback")
        if isinstance(fb, str) and fb.strip():
            return fb.strip()
    return None


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("type") or m.get("role") or "message"
        content = m.get("content")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()


def _extract_plan(summary_item: dict[str, Any]) -> list[str] | None:
    plan = summary_item.get("plan")
    if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
        return plan
    return None


def _extract_executed_workflow(raw: dict[str, Any]) -> list[str] | None:
    """Infer executed tool order from output.tool_outputs keys like step_0_search_mofs."""
    output = raw.get("output")
    if not isinstance(output, dict):
        return None
    tool_outputs = output.get("tool_outputs")
    if not isinstance(tool_outputs, dict):
        return None

    steps: list[tuple[int, str]] = []
    for k in tool_outputs.keys():
        if not isinstance(k, str):
            continue
        m = re.match(r"^step_(\d+)_([a-z_]+)$", k)
        if not m:
            continue
        idx = int(m.group(1))
        tool = m.group(2)
        steps.append((idx, tool))

    if not steps:
        return []
    return [tool for _, tool in sorted(steps, key=lambda x: x[0])]


def _parse_expected_plan(expectation: str) -> list[str] | None:
    # Try bracket list first: [search_mofs, optimize_structure]
    m = re.search(r"\[(.*?)\]", expectation)
    if m:
        inner = m.group(1)
        tools = [t.strip() for t in inner.split(",")]
        tools = [t for t in tools if t in TOOLS]
        if tools:
            return tools

    # Try arrow sequence: search_mofs -> optimize_structure -> calculate_energy
    if "->" in expectation:
        found = re.findall(r"\b(?:search_mofs|optimize_structure|calculate_energy)\b", expectation)
        if found:
            return found

    # If expectation explicitly says should be out_of_scope / need_context
    if re.search(r"\b(out_of_scope|need_context)\b", expectation):
        return []

    # Otherwise, cannot infer a concrete plan
    return None


def _looks_like_clarification(text: str) -> bool:
    if not text:
        return False
    # Heuristic: question marks or typical request-for-info phrases.
    return (
        "?" in text
        or "please provide" in text.lower()
        or "i need" in text.lower()
        or "do you have" in text.lower()
        or "could you" in text.lower()
    )


def _check_case(expectation: str, actual_plan: list[str] | None, approved: Any, raw: dict[str, Any]) -> CheckResult:
    if "error" in raw:
        return CheckResult(False, "server_error", None)

    expected_plan = _parse_expected_plan(expectation)

    # If we can infer an expected plan (non-empty), compare directly.
    if expected_plan is not None and expected_plan != []:
        if actual_plan == expected_plan and approved is True:
            return CheckResult(True, "plan_matches_and_approved", expected_plan)
        if actual_plan != expected_plan:
            return CheckResult(False, "plan_mismatch", expected_plan)
        if approved is not True:
            return CheckResult(False, "not_approved", expected_plan)
        return CheckResult(None, "manual_review", expected_plan)

    # Expectations about need_context/out_of_scope.
    if expected_plan == []:
        # The contract expects analyzer to emit JSON for need_context/out_of_scope.
        # In practice, we treat 'plan=[] and approved=False' as 'ended early'.
        # But we also verify it likely asked for clarification.
        output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
        last_ai = None
        msgs = output.get("messages") if isinstance(output, dict) else None
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("type") == "ai":
                    c = m.get("content")
                    if isinstance(c, str):
                        last_ai = c
                        break
        if actual_plan == [] and approved is False and isinstance(last_ai, str) and _looks_like_clarification(last_ai):
            # Workflow-wise this is acceptable (it asked for missing info), but it violates the analyzer's JSON-only contract.
            return CheckResult(True, "asked_for_context_but_not_json_contract", expected_plan)
        if actual_plan == [] and approved is False:
            return CheckResult(None, "ended_without_plan_manual_review", expected_plan)
        return CheckResult(False, "unexpected_plan_or_approval", expected_plan)

    # No concrete expectation: mark manual review, but still record if it planned and approved.
    if actual_plan and approved is True:
        return CheckResult(None, "no_concrete_expectation_but_planned", expected_plan)
    if actual_plan == []:
        return CheckResult(None, "no_concrete_expectation_and_no_plan", expected_plan)
    return CheckResult(None, "manual_review", expected_plan)


def _check_against_desired(
    desired: list[str] | None,
    acceptable: list[list[str]] | None,
    actual_plan: list[str] | None,
    approved: Any,
    raw: dict[str, Any],
) -> CheckResult | None:
    """If desired/acceptable workflows are provided, use them for an explicit pass/fail."""
    if desired is None and not acceptable:
        return None
    if "error" in raw:
        return CheckResult(False, "server_error", desired)

    # desired = [] means the intended behavior is to ask for context / refuse / end.
    if desired == []:
        output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
        last_ai = None
        msgs = output.get("messages") if isinstance(output, dict) else None
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("type") == "ai":
                    c = m.get("content")
                    if isinstance(c, str):
                        last_ai = c
                        break
        if actual_plan == [] and approved is False and isinstance(last_ai, str) and _looks_like_clarification(last_ai):
            # Workflow-wise this is acceptable (it asked for missing info), but it violates the analyzer's JSON-only contract.
            return CheckResult(True, "asked_for_context_but_not_json_contract", desired)
        if actual_plan == []:
            return CheckResult(True, "ended_without_plan_as_desired", desired)
        return CheckResult(False, "unexpected_plan_when_should_end", desired)

    acceptable_plans = []
    if acceptable:
        acceptable_plans.extend([p for p in acceptable if isinstance(p, list)])
    if desired is not None:
        acceptable_plans.append(desired)

    # If we got no plan (None), that is a failure for cases expecting a workflow.
    if actual_plan is None:
        return CheckResult(False, "missing_plan", desired)

    if any(actual_plan == p for p in acceptable_plans) and approved is True:
        return CheckResult(True, "plan_matches_desired_and_approved", desired)

    if any(actual_plan == p for p in acceptable_plans) and approved is not True:
        return CheckResult(False, "plan_matches_but_not_approved", desired)

    return CheckResult(False, "plan_mismatch", desired)


def _md_escape(text: str) -> str:
    # Minimal escaping for tables.
    return text.replace("|", "\\|")


def _one_line(text: str) -> str:
    """Normalize whitespace so the summary stays readable."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: render_eval_report.py <run_dir>")

    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    summary = _read_json(summary_path)
    if not isinstance(summary, list):
        raise SystemExit("summary.json must be a list")

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    md_lines: list[str] = []
    md_lines.append(f"# Analyzer Evaluation Report\n")
    md_lines.append(f"Run directory: {run_dir}\n")

    md_lines.append("## Scoreboard")
    md_lines.append("| Case | Meets Expectation | Approved | Expected Plan | Actual Plan | Reason | Raw |")
    md_lines.append("|---|---:|---:|---|---|---|---|")

    for item in summary:
        raw_path = Path(item["raw"])
        raw = _read_json(raw_path) if raw_path.exists() else {"error": "missing_raw_file"}

        case_id = item.get("case_id")
        title = item.get("title")
        expectation = item.get("expectation") or ""
        approved = item.get("approved")
        actual_plan = _extract_plan(item)

        desired = _normalize_plan(item.get("desired_workflow"))
        acceptable = item.get("acceptable_workflows")
        acceptable_norm: list[list[str]] | None = None
        if isinstance(acceptable, list):
            tmp: list[list[str]] = []
            for p in acceptable:
                pn = _normalize_plan(p)
                if pn is not None:
                    tmp.append(pn)
            acceptable_norm = tmp if tmp else None

        check = _check_against_desired(desired, acceptable_norm, actual_plan, approved, raw)
        if check is None:
            check = _check_case(expectation, actual_plan, approved, raw)

        executed_workflow = _extract_executed_workflow(raw)

        meets_str = "MANUAL" if check.meets is None else ("YES" if check.meets else "NO")
        expected_plan_str = "" if check.expected_plan is None else str(check.expected_plan)
        actual_plan_str = "" if actual_plan is None else str(actual_plan)

        md_lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(_one_line(f"{case_id}: {title}")),
                    meets_str,
                    _md_escape(str(approved)),
                    _md_escape(expected_plan_str),
                    _md_escape(actual_plan_str),
                    _md_escape(check.reason),
                    f"[{raw_path.name}]({raw_path.name})",
                ]
            )
            + " |"
        )

        messages_text = _messages_to_text(_extract_messages(raw))
        questions = _extract_human_questions(raw)
        supervisor_feedback = _extract_supervisor_feedback(raw)

        results.append(
            {
                "case_id": case_id,
                "title": title,
                "expectation": expectation,
                "desired_workflow": desired,
                "acceptable_workflows": acceptable_norm,
                "expected_plan": check.expected_plan,
                "actual_plan": actual_plan,
                "executed_workflow": executed_workflow,
                "approved": approved,
                "meets_expectation": check.meets,
                "reason": check.reason,
                "raw": str(raw_path),
                "messages": messages_text,
                "questions": questions,
                "supervisor_feedback": supervisor_feedback,
            }
        )

        summary_rows.append(
            {
                "case_id": case_id,
                "title": title,
                "questions": questions,
                "expectation": expectation,
                "desired_workflow": desired,
                "acceptable_workflows": acceptable_norm,
                "expected_plan": check.expected_plan,
                "actual_plan": actual_plan,
                "executed_workflow": executed_workflow,
                "approved": approved,
                "meets_expectation": check.meets,
                "reason": check.reason,
                "supervisor_feedback": supervisor_feedback,
                "raw_file": raw_path.name,
            }
        )

    md_lines.append("\n## Details")
    for r in results:
        md_lines.append(f"\n### {r['case_id']}: {r['title']}")
        md_lines.append(f"- Meets expectation: {r['meets_expectation']} ({r['reason']})")
        md_lines.append(f"- Approved: {r['approved']}")
        md_lines.append(f"- Expected plan: {r['expected_plan']}")
        md_lines.append(f"- Desired workflow: {r.get('desired_workflow')}")
        md_lines.append(f"- Acceptable workflows: {r.get('acceptable_workflows')}")
        md_lines.append(f"- Actual plan: {r['actual_plan']}")
        md_lines.append(f"- Executed workflow: {r.get('executed_workflow')}")
        md_lines.append(f"- Raw: {Path(r['raw']).name}")
        md_lines.append("\n**Conversation**\n")
        md_lines.append("```\n" + (r["messages"] or "<no messages>") + "\n```")
        md_lines.append("\n**Expectation**\n")
        md_lines.append("```\n" + (r["expectation"] or "") + "\n```")

    report_md = run_dir / "report.md"
    report_json = run_dir / "report.json"
    report_summary_md = run_dir / "report_summary.md"
    workflow_table_md = run_dir / "workflow_table.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write a cleaner, compact summary report.
    sl: list[str] = []
    sl.append("# Analyzer Eval (Clear Summary)\n")
    sl.append(f"Run directory: {run_dir}\n")
    sl.append("This file is intended to be quickly scannable: question(s), expectation, expected workflow, actual workflow, approval, and whether it met the expected workflow.")

    for row in summary_rows:
        sl.append("\n---\n")
        sl.append(f"## {row['case_id']}: {row['title']}")
        qs = row.get("questions") or []
        if qs:
            sl.append("**Question(s)**")
            for q in qs:
                sl.append(f"- {_one_line(q)}")
        else:
            sl.append("**Question(s)**")
            sl.append("- <not available in output>")

        sl.append("\n**Expectation**")
        sl.append(f"- {_one_line(row.get('expectation') or '')}")

        sl.append("\n**Expected Workflow**")
        sl.append(f"- {row.get('expected_plan')}")

        sl.append("\n**Desired Workflow (from test metadata)**")
        sl.append(f"- {row.get('desired_workflow')}")
        acc = row.get("acceptable_workflows")
        if acc:
            sl.append("\n**Acceptable Workflows (also OK)**")
            sl.append(f"- {acc}")

        sl.append("\n**Actual Workflow**")
        sl.append(f"- plan: {row.get('actual_plan')}")
        sl.append(f"- approved: {row.get('approved')}")

        sl.append("\n**Executed Workflow (from tool outputs)**")
        sl.append(f"- {row.get('executed_workflow')}")

        sl.append("\n**Meets Expected Workflow**")
        sl.append(f"- {row.get('meets_expectation')} ({row.get('reason')})")

        fb = row.get("supervisor_feedback")
        if fb:
            sl.append("\n**Supervisor Feedback**")
            sl.append(f"- {_one_line(fb)[:900]}" + ("…" if len(_one_line(fb)) > 900 else ""))

        sl.append("\n**Raw Output**")
        sl.append(f"- {row.get('raw_file')}")

    report_summary_md.write_text("\n".join(sl) + "\n", encoding="utf-8")

    # Extra-compact table (one row per case).
    tl: list[str] = []
    tl.append("# Analyzer Eval (Workflow Table)\n")
    tl.append(f"Run directory: {run_dir}\n")
    tl.append(
        "| case_id | question | desired_workflow | plan | executed_workflow | approved | meets | reason | raw |"
    )
    tl.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        qs = row.get("questions") or []
        q0 = qs[0] if qs else ""
        tl.append(
            "| {case_id} | {q} | {desired} | {plan} | {executed} | {approved} | {meets} | {reason} | {raw} |".format(
                case_id=row.get("case_id"),
                q=_one_line(q0)[:140],
                desired=str(row.get("desired_workflow")),
                plan=str(row.get("actual_plan")),
                executed=str(row.get("executed_workflow")),
                approved=str(row.get("approved")),
                meets=str(row.get("meets_expectation")),
                reason=str(row.get("reason")),
                raw=str(row.get("raw_file")),
            )
        )
    workflow_table_md.write_text("\n".join(tl) + "\n", encoding="utf-8")

    print(f"Wrote {report_md}")
    print(f"Wrote {report_json}")
    print(f"Wrote {report_summary_md}")
    print(f"Wrote {workflow_table_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
