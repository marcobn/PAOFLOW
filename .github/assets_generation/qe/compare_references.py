from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.integration.qe.jobs import discover_jobs

NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


@dataclass
class FileComparison:
    relpath: str
    status: str
    reason: str | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    values_compared: int = 0


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _extract_numbers(text: str) -> list[float]:
    return [float(m.group(0)) for m in NUM_RE.finditer(text)]


def _compare_numeric(a_vals: list[float], b_vals: list[float], *, atol: float, rtol: float) -> tuple[bool, float, float]:
    max_abs = 0.0
    max_rel = 0.0
    ok = True
    for av, bv in zip(a_vals, b_vals):
        ad = abs(av - bv)
        denom = max(abs(av), abs(bv), 1e-300)
        rd = ad / denom
        if ad > max_abs:
            max_abs = ad
        if rd > max_rel:
            max_rel = rd
        if ad > (atol + rtol * abs(av)):
            ok = False
    return ok, max_abs, max_rel


def _compare_file(ref_file: Path, cand_file: Path, *, atol: float, rtol: float) -> FileComparison:
    relpath = ref_file.name

    if not cand_file.exists():
        return FileComparison(relpath=relpath, status="missing_candidate", reason="candidate file missing")

    ref_text = _read_text(ref_file)
    cand_text = _read_text(cand_file)

    if ref_text is None or cand_text is None:
        if ref_file.read_bytes() == cand_file.read_bytes():
            return FileComparison(relpath=relpath, status="exact_binary")
        return FileComparison(relpath=relpath, status="different_binary", reason="binary files differ")

    ref_nums = _extract_numbers(ref_text)
    cand_nums = _extract_numbers(cand_text)

    if ref_nums and cand_nums and len(ref_nums) == len(cand_nums):
        ok, max_abs, max_rel = _compare_numeric(ref_nums, cand_nums, atol=atol, rtol=rtol)
        return FileComparison(
            relpath=relpath,
            status="numeric_match" if ok else "numeric_mismatch",
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            values_compared=len(ref_nums),
        )

    if ref_text == cand_text:
        return FileComparison(relpath=relpath, status="exact_text")

    reason = "text differs"
    if (ref_nums or cand_nums) and len(ref_nums) != len(cand_nums):
        reason = "different numeric token counts"
    return FileComparison(relpath=relpath, status="text_mismatch", reason=reason)


def _collect_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file()])


def compare_job(job_dir: Path, baseline_dirname: str, candidate_dirname: str, *, atol: float, rtol: float) -> dict[str, Any]:
    baseline = job_dir / baseline_dirname
    candidate = job_dir / candidate_dirname

    result: dict[str, Any] = {
        "job_dir": str(job_dir),
        "baseline": baseline_dirname,
        "candidate": candidate_dirname,
        "status": "ok",
        "files": [],
        "summary": {
            "total": 0,
            "ok": 0,
            "mismatch": 0,
            "missing_candidate": 0,
            "extra_candidate": 0,
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
        },
    }

    if not baseline.is_dir():
        result["status"] = "missing_baseline_dir"
        return result

    if not candidate.is_dir():
        result["status"] = "missing_candidate_dir"
        return result

    baseline_files = _collect_files(baseline)
    candidate_files = _collect_files(candidate)
    baseline_rel = {p.relative_to(baseline).as_posix(): p for p in baseline_files}
    candidate_rel = {p.relative_to(candidate).as_posix(): p for p in candidate_files}

    all_paths = sorted(set(baseline_rel.keys()) | set(candidate_rel.keys()))

    for rel in all_paths:
        ref_file = baseline_rel.get(rel)
        cand_file = candidate_rel.get(rel)
        if ref_file is None:
            result["files"].append({
                "relpath": rel,
                "status": "extra_candidate",
                "reason": "file exists only in candidate",
            })
            result["summary"]["extra_candidate"] += 1
            continue

        cmp_result = _compare_file(ref_file, cand_file if cand_file is not None else candidate / rel, atol=atol, rtol=rtol)
        rec = {
            "relpath": rel,
            "status": cmp_result.status,
            "reason": cmp_result.reason,
            "max_abs_diff": cmp_result.max_abs_diff,
            "max_rel_diff": cmp_result.max_rel_diff,
            "values_compared": cmp_result.values_compared,
        }
        result["files"].append(rec)
        result["summary"]["total"] += 1

        if cmp_result.status in {"numeric_match", "exact_text", "exact_binary"}:
            result["summary"]["ok"] += 1
        elif cmp_result.status == "missing_candidate":
            result["summary"]["missing_candidate"] += 1
        else:
            result["summary"]["mismatch"] += 1

        if cmp_result.max_abs_diff is not None:
            result["summary"]["max_abs_diff"] = max(result["summary"]["max_abs_diff"], cmp_result.max_abs_diff)
        if cmp_result.max_rel_diff is not None:
            result["summary"]["max_rel_diff"] = max(result["summary"]["max_rel_diff"], cmp_result.max_rel_diff)

    if result["summary"]["mismatch"] or result["summary"]["missing_candidate"] or result["summary"]["extra_candidate"]:
        result["status"] = "mismatch"

    return result


def write_markdown_report(report: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# PAOFLOW Reference Comparison")
    lines.append("")
    lines.append(f"- Baseline: `{report['baseline_dirname']}`")
    lines.append(f"- Candidate: `{report['candidate_dirname']}`")
    lines.append(f"- atol: `{report['atol']}`")
    lines.append(f"- rtol: `{report['rtol']}`")
    lines.append("")

    total_jobs = len(report["jobs"])
    mismatched_jobs = sum(1 for j in report["jobs"] if j["status"] != "ok")
    lines.append(f"## Summary")
    lines.append(f"- Jobs compared: `{total_jobs}`")
    lines.append(f"- Jobs with mismatches/errors: `{mismatched_jobs}`")
    lines.append("")

    for job in report["jobs"]:
        lines.append(f"## {job['job_id']}")
        lines.append(f"- Status: `{job['status']}`")
        if job["status"] in {"missing_baseline_dir", "missing_candidate_dir"}:
            lines.append("")
            continue
        s = job["summary"]
        lines.append(f"- Files checked: `{s['total']}`")
        lines.append(f"- OK: `{s['ok']}`")
        lines.append(f"- Mismatch: `{s['mismatch']}`")
        lines.append(f"- Missing in candidate: `{s['missing_candidate']}`")
        lines.append(f"- Extra in candidate: `{s['extra_candidate']}`")
        lines.append(f"- Max abs diff: `{s['max_abs_diff']:.6e}`")
        lines.append(f"- Max rel diff: `{s['max_rel_diff']:.6e}`")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notebook(report_json: Path, out_nb: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    "# PAOFLOW Reference Comparison Analysis\n",
                    "Load the JSON report and visualize mismatch counts and numeric diff distributions.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "\n",
                    f"report_path = Path(r'{str(report_json)}')\n",
                    "report = json.loads(report_path.read_text(encoding='utf-8'))\n",
                    "print('Loaded report:', report_path)\n",
                    "print('Jobs:', len(report['jobs']))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "job_ids = [j['job_id'] for j in report['jobs']]\n",
                    "mismatches = [j.get('summary', {}).get('mismatch', 0) + j.get('summary', {}).get('missing_candidate', 0) + j.get('summary', {}).get('extra_candidate', 0) for j in report['jobs']]\n",
                    "\n",
                    "plt.figure(figsize=(10, 4))\n",
                    "plt.bar(range(len(job_ids)), mismatches)\n",
                    "plt.xticks(range(len(job_ids)), job_ids, rotation=70, ha='right')\n",
                    "plt.ylabel('Mismatch/Missing/Extra count')\n",
                    "plt.title('Per-job differences')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"language": "python"},
                "outputs": [],
                "source": [
                    "abs_diffs = []\n",
                    "rel_diffs = []\n",
                    "for job in report['jobs']:\n",
                    "    for f in job.get('files', []):\n",
                    "        if f.get('max_abs_diff') is not None:\n",
                    "            abs_diffs.append(f['max_abs_diff'])\n",
                    "        if f.get('max_rel_diff') is not None:\n",
                    "            rel_diffs.append(f['max_rel_diff'])\n",
                    "\n",
                    "fig, ax = plt.subplots(1, 2, figsize=(10, 4))\n",
                    "ax[0].hist(abs_diffs, bins=40)\n",
                    "ax[0].set_title('Absolute diff distribution')\n",
                    "ax[0].set_xlabel('abs diff')\n",
                    "ax[0].set_ylabel('count')\n",
                    "\n",
                    "ax[1].hist(rel_diffs, bins=40)\n",
                    "ax[1].set_title('Relative diff distribution')\n",
                    "ax[1].set_xlabel('rel diff')\n",
                    "ax[1].set_ylabel('count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_nb.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PAOFLOW Reference folders across examples/jobs.")
    parser.add_argument("--qe-root", type=Path, default=repo_root / "examples" / "qe_examples")
    parser.add_argument("--baseline", type=str, default="Reference")
    parser.add_argument("--candidate", type=str, default="Reference2")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--notebook-out", type=Path)
    args = parser.parse_args()

    qe_root = args.qe_root.resolve()
    jobs = discover_jobs(qe_root)

    report: dict[str, Any] = {
        "qe_root": str(qe_root),
        "baseline_dirname": args.baseline,
        "candidate_dirname": args.candidate,
        "atol": args.atol,
        "rtol": args.rtol,
        "jobs": [],
    }

    for job in jobs:
        rec = compare_job(
            job.job_dir,
            args.baseline,
            args.candidate,
            atol=args.atol,
            rtol=args.rtol,
        )
        rec["job_id"] = job.id
        report["jobs"].append(rec)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_out = args.md_out if args.md_out is not None else args.json_out.with_suffix(".md")
    write_markdown_report(report, md_out)

    if args.notebook_out is not None:
        args.notebook_out.parent.mkdir(parents=True, exist_ok=True)
        write_notebook(args.json_out.resolve(), args.notebook_out)

    has_diff = False
    for job in report["jobs"]:
        if job["status"] != "ok":
            has_diff = True
            break
    raise SystemExit(1 if has_diff else 0)


if __name__ == "__main__":
    main()
