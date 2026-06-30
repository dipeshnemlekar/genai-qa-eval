"""
runners/build_dashboard.py
==========================
Scan all historical evaluation CSVs and produce a single, fully self-contained
offline HTML dashboard at  reports/dashboard.html.

Usage
-----
    python runners/build_dashboard.py [options]

Options
    --reports-dir PATH   Root of the reports tree  (default: <repo>/reports)
    --out         PATH   Output HTML file           (default: <reports-dir>/dashboard.html)
    --last        N      Limit trend chart to the N most-recent runs
    --open               Open the file in the default browser after building

Exit codes
    0  success
    1  no CSV files found, or fatal error

Real CSV columns (as confirmed from existing reports):
    TC No | Input | Expected Output | Actual Output | Judge Score | LLM Reason | Human Score | Human Reason

Column mapping:
    case      <- "TC No"
    score     <- "Judge Score"  (1=pass, 0=fail, empty=Error)
    reason    <- "LLM Reason"
    threshold <- not in CSV; falls back to eval_config.THRESHOLDS
    passed    <- derived: score>=1 -> True, score==0 -> False, empty -> None (Error)
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import eval_config as cfg
except ImportError:
    class _Cfg:
        CATEGORY_LABELS    = {"rag": "RAG", "safety": "Safety", "llm": "LLM Quality", "adversarial": "Security Posture"}
        METRIC_CATEGORY    = {}
        LOWER_IS_BETTER    = {"Hallucination", "Pii Leakage", "Bias", "Toxicity"}
        THRESHOLDS         = {}
        REGRESSION_DELTA   = 0.1
        RUN_BUCKET_SECONDS = 120
    cfg = _Cfg()

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
except ImportError:
    print("[ERROR] plotly is required:  pip install plotly", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAMP_RE  = re.compile(r"_report_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$")
METRIC_RE = re.compile(r"^(.+?)_report_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv$")

# Ordered lists — first match wins. Put highest-priority aliases first.
# For 'case', 'tc no' must come before 'input' to pick up the test ID column
# (e.g. ECOM_RAG_001) rather than the question text in the Input column.
ALIASES: dict[str, list[str]] = {
    "case":      ["tc no", "test case", "case", "name", "test_case", "test_name", "input"],
    "score":     ["judge score", "score", "metric_score", "value"],
    "threshold": ["threshold", "min_score"],
    "passed":    ["passed", "success", "result", "status"],
    "reason":    ["llm reason", "reason", "explanation", "judge_reason"],
    "human_score": ["human score"],
}
REQUIRED_ALIASES = {"case", "score"}

GREEN  = "#2ecc71"
RED    = "#e74c3c"
GREY   = "#95a5a6"
AMBER  = "#f39c12"
LINE_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
]

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _resolve_aliases(fieldnames: list[str]) -> dict[str, str | None]:
    """Map normalised field names to the actual column header found in the CSV."""
    lower_map = {f.strip().lower(): f for f in (fieldnames or [])}
    result: dict[str, str | None] = {}
    for norm, candidates in ALIASES.items():
        hit = next((lower_map[c] for c in candidates if c in lower_map), None)
        result[norm] = hit
    missing = [k for k in REQUIRED_ALIASES if not result.get(k)]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found. "
            f"Available headers: {list(fieldnames or [])}"
        )
    return result


def _to_float(v: Any) -> float | None:
    s = str(v).strip().lower() if v is not None else ""
    if s in ("", "none", "null", "na", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _derive_passed(score: float | None) -> bool | None:
    """
    Infer pass/fail from judge score.
    score=1.0 -> True (Passed)
    score=0.0 -> False (Failed)
    score=None -> None (Error — judge returned no score)
    """
    if score is None:
        return None
    return score >= 1.0


def _to_bool_from_col(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "pass", "passed", "yes", "success"}


def _slug_to_name(slug: str) -> str:
    """answer_relevancy -> Answer Relevancy"""
    return slug.replace("_", " ").title()


def _category_display(folder: str, metric: str) -> str:
    """Return display category: folder label takes priority over metric mapping."""
    label = cfg.CATEGORY_LABELS.get(folder.lower())
    if label:
        return label
    return cfg.METRIC_CATEGORY.get(metric, "Other")


def load_csv(path: Path) -> list[dict]:
    """Parse one metric CSV; return list of row dicts with normalised keys."""
    nm = METRIC_RE.match(path.name)
    sm = STAMP_RE.search(path.name)
    if not nm or not sm:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    metric    = _slug_to_name(nm.group(1))
    run_stamp = datetime.strptime(sm.group(1), "%Y-%m-%d_%H-%M-%S")
    folder    = path.parent.name.lower()
    category  = _category_display(folder, metric)

    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        a = _resolve_aliases(reader.fieldnames)
        for raw in reader:
            case_val = (raw.get(a["case"]) or "").strip()
            if not case_val:
                continue

            score      = _to_float(raw.get(a["score"])) if a.get("score") else None
            reason     = (raw.get(a["reason"]) or "").strip() if a.get("reason") else ""
            thr_raw    = raw.get(a["threshold"]) if a.get("threshold") else None
            threshold  = _to_float(thr_raw) if thr_raw else cfg.THRESHOLDS.get(metric)
            human_score = _to_float(raw.get(a["human_score"])) if a.get("human_score") else None

            # Determine pass/fail strictly
            if a.get("passed") and str(raw.get(a["passed"]) or "").strip():
                passed = _to_bool_from_col(raw.get(a["passed"]))
            else:
                passed = _derive_passed(score)

            rows.append({
                "case":      case_val[:120],
                "metric":    metric,
                "category":  category,
                "score":     score,
                "threshold": threshold,
                "passed":    passed,
                "reason":    reason,
                "human_score": human_score,
                "run_id":    run_stamp,
            })
    return rows


def load_all(reports_dir: Path) -> list[dict]:
    """Recursively scan reports_dir for metric CSVs and aggregate all rows."""
    all_rows: list[dict] = []
    for csv_path in sorted(reports_dir.rglob("*.csv")):
        if "_report_" not in csv_path.name:
            continue
        try:
            all_rows.extend(load_csv(csv_path))
        except Exception as exc:
            print(f"[warn] Skipping {csv_path.name}: {exc}")
    return all_rows


# ---------------------------------------------------------------------------
# Run grouping
# ---------------------------------------------------------------------------

def group_runs(
    rows: list[dict],
    bucket_seconds: int = 120,
) -> list[tuple[str, list[dict]]]:
    """
    Return ordered list of (run_label, rows_in_run), earliest first.
    Timestamps within bucket_seconds of each other are merged into one run.
    """
    all_stamps = sorted({r["run_id"] for r in rows})
    if not all_stamps:
        return []

    buckets: list[list[datetime]] = []
    current: list[datetime] = [all_stamps[0]]
    for stamp in all_stamps[1:]:
        if (stamp - current[-1]).total_seconds() <= bucket_seconds:
            current.append(stamp)
        else:
            buckets.append(current)
            current = [stamp]
    buckets.append(current)

    result = []
    for bucket in buckets:
        bs    = set(bucket)
        label = max(bucket).strftime("%Y-%m-%d %H:%M")
        result.append((label, [r for r in rows if r["run_id"] in bs]))
    return result


def _state(row: dict) -> str:
    if row["passed"] is None:
        return "Error"
    return "Passed" if row["passed"] else "Failed"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_overall_pie(latest_rows: list[dict]) -> str:
    """Chart - Overall Pie: Passed, Failed, Error."""
    counts = {"Passed": 0, "Failed": 0, "Error": 0}
    for row in latest_rows:
        counts[_state(row)] += 1
        
    labels = ["Passed", "Failed", "Error"]
    values = [counts["Passed"], counts["Failed"], counts["Error"]]
    colors = [GREEN, RED, GREY]
    
    # Filter out 0 counts to make it cleaner
    f_labels, f_values, f_colors = [], [], []
    for l, v, c in zip(labels, values, colors):
        if v > 0:
            f_labels.append(l)
            f_values.append(v)
            f_colors.append(c)
    
    fig = go.Figure(go.Pie(
        labels=f_labels, values=f_values,
        marker=dict(colors=f_colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent", hole=0.4,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Overall Status (Latest)", font=dict(size=18)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=60, b=20), height=350, showlegend=True,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def chart_category_sunburst(latest_rows: list[dict]) -> str:
    """Chart - Sunburst: Category -> Metric breakdown."""
    cat_counts = defaultdict(int)
    metric_counts = defaultdict(lambda: defaultdict(int))
    
    for row in latest_rows:
        cat = row["category"]
        metric = row["metric"]
        cat_counts[cat] += 1
        metric_counts[cat][metric] += 1
        
    ids = ["All"]
    labels = ["All Tests"]
    parents = [""]
    values = [len(latest_rows)]
    
    for cat, count in cat_counts.items():
        ids.append(cat)
        labels.append(cat)
        parents.append("All")
        values.append(count)
        
    for cat, metrics in metric_counts.items():
        for metric, count in metrics.items():
            ids.append(f"{cat}-{metric}")
            labels.append(metric)
            parents.append(cat)
            values.append(count)
            
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percentParent} of Parent<extra></extra>",
        marker=dict(line=dict(color="#1a1a2e", width=1))
    ))
    
    fig.update_layout(
        title=dict(text="Test Distribution by Category", font=dict(size=18)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=60, b=20), height=350,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def chart_pass_fail(latest_rows: list[dict]) -> str:
    """Chart 1 - Horizontal Stacked bar: Passed/Failed/Error counts per metric, sorted by pass rate."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"Passed": 0, "Failed": 0, "Error": 0, "Total": 0}
    )
    for row in latest_rows:
        state = _state(row)
        counts[row["metric"]][state] += 1
        counts[row["metric"]]["Total"] += 1

    # Sort metrics by pass rate (ascending so highest is at top in Plotly horizontal bar)
    metrics_sorted = sorted(
        counts.keys(),
        key=lambda m: (counts[m]["Passed"] / counts[m]["Total"]) if counts[m]["Total"] else 0
    )

    fig = go.Figure()
    # Add traces in reverse order to stack them correctly
    fig.add_trace(go.Bar(name="Error", y=metrics_sorted, x=[counts[m]["Error"] for m in metrics_sorted],
                         orientation='h', marker_color=GREY,
                         text=[counts[m]["Error"] if counts[m]["Error"] else "" for m in metrics_sorted], textposition="inside"))
    fig.add_trace(go.Bar(name="Failed", y=metrics_sorted, x=[counts[m]["Failed"] for m in metrics_sorted],
                         orientation='h', marker_color=RED,
                         text=[counts[m]["Failed"] if counts[m]["Failed"] else "" for m in metrics_sorted], textposition="inside"))
    fig.add_trace(go.Bar(name="Passed", y=metrics_sorted, x=[counts[m]["Passed"] for m in metrics_sorted],
                         orientation='h', marker_color=GREEN,
                         text=[counts[m]["Passed"] if counts[m]["Passed"] else "" for m in metrics_sorted], textposition="inside"))
    
    fig.update_layout(
        barmode="stack",
        title=dict(text="Per-Metric Pass / Fail Snapshot (Latest)", font=dict(size=18)),
        xaxis_title="Count", yaxis_title="Metric",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=-0.2),
        margin=dict(l=150, r=20, t=60, b=80), height=420,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def chart_trend(
    run_groups: list[tuple[str, list[dict]]],
    last_n: int | None,
) -> str:
    """Chart 2 - Line: pass-rate trend per metric + Overall across runs."""
    groups = run_groups[-last_n:] if last_n else run_groups
    all_metrics = sorted({r["metric"] for _, rw in groups for r in rw})
    labels = [lb for lb, _ in groups]
    fig = go.Figure()

    for idx, metric in enumerate(all_metrics):
        rates: list[float | None] = []
        for _, rrows in groups:
            m = [r for r in rrows if r["metric"] == metric]
            rates.append(sum(1 for r in m if r["passed"] is True) / len(m) if m else None)
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        fig.add_trace(go.Scatter(
            x=labels, y=rates, name=metric, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7),
            connectgaps=False,
        ))

    overall: list[float | None] = [
        sum(1 for r in rrows if r["passed"] is True) / len(rrows) if rrows else None
        for _, rrows in groups
    ]
    fig.add_trace(go.Scatter(
        x=labels, y=overall, name="Overall", mode="lines+markers",
        line=dict(color="#ffffff", width=3, dash="dash"),
        marker=dict(size=9, color="#ffffff"), connectgaps=False,
    ))

    fig.update_layout(
        title=dict(text="Metric Pass-Rate History", font=dict(size=18)),
        xaxis_title="Date / Run",
        yaxis=dict(title="Pass Rate", tickformat=".0%", range=[0, 1.05],
                   gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=-0.35),
        margin=dict(l=60, r=20, t=60, b=150), height=440,
    )
    fig.update_xaxes(tickangle=-30, gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def chart_category_trend(run_groups: list[tuple[str, list[dict]]], last_n: int | None) -> str:
    """Chart 3 - Area: pass-rate trend per category over time."""
    groups = run_groups[-last_n:] if last_n else run_groups
    all_categories = sorted({r["category"] for _, rw in groups for r in rw})
    labels = [lb for lb, _ in groups]
    fig = go.Figure()

    for idx, category in enumerate(all_categories):
        rates: list[float | None] = []
        for _, rrows in groups:
            c = [r for r in rrows if r["category"] == category]
            rates.append(sum(1 for r in c if r["passed"] is True) / len(c) if c else None)
        color = LINE_COLORS[(idx + 4) % len(LINE_COLORS)] # offset colors from metric lines
        fig.add_trace(go.Scatter(
            x=labels, y=rates, name=category, mode="lines+markers",
            fill="tozeroy", line=dict(color=color, width=2),
            marker=dict(size=6), connectgaps=False,
        ))

    fig.update_layout(
        title=dict(text="Category Pass-Rate Trend", font=dict(size=18)),
        xaxis_title="Date / Run",
        yaxis=dict(title="Pass Rate", tickformat=".0%", range=[0, 1.05],
                   gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=20, t=60, b=100), height=420,
    )
    fig.update_xaxes(tickangle=-30, gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def chart_test_case_heatmap(run_groups: list[tuple[str, list[dict]]], last_n: int | None) -> str:
    """Chart 4 - Heatmap: Pass/Fail state of individual test cases over time."""
    groups = run_groups[-last_n:] if last_n else run_groups
    labels = [lb for lb, _ in groups]
    
    # Collect all test cases and their histories
    case_history: dict[str, list[int | None]] = defaultdict(lambda: [None] * len(labels))
    for t_idx, (_, rrows) in enumerate(groups):
        for r in rrows:
            state_val = 1 if r["passed"] is True else (0 if r["passed"] is False else None)
            case_history[r["case"]][t_idx] = state_val
            
    cases = sorted(case_history.keys(), reverse=True) # Y axis (reverse to keep A-Z top down)
    z_data = [case_history[c] for c in cases]
    
    colorscale = [
        [0.0, RED],
        [0.5, GREY], # shouldn't hit 0.5 but required for continuous scale fallback
        [1.0, GREEN]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=labels, y=cases,
        colorscale=colorscale, zmin=0, zmax=1,
        showscale=False,
        hoverongaps=False,
        hovertemplate="Run: %{x}<br>Case: %{y}<br>Passed: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Test Case Failure Heatmap", font=dict(size=18)),
        xaxis_title="Date / Run",
        yaxis_title="Test Case",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=150, r=20, t=60, b=120), height=max(400, len(cases) * 18 + 150),
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(tickmode="linear", automargin=True)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Regression diff
# ---------------------------------------------------------------------------

def compute_diff(
    prev_rows: list[dict],
    latest_rows: list[dict],
    delta: float = 0.1,
) -> tuple[list, list, list]:
    """
    Return (newly_failing, newly_passing, score_drops) pairs.

    Error (passed=None) is a DISTINCT state — it never counts as 'failing'
    in the regression diff. It has its own Error table below the diff.
    Only explicit True/False transitions are tracked here.
    """
    key      = lambda r: (r["case"], r["metric"])
    prev_map = {key(r): r for r in prev_rows}
    nf, np_, drops = [], [], []

    for r in latest_rows:
        p = prev_map.get(key(r))
        if not p:
            continue
        pp = p["passed"]   # True | False | None (Error)
        rp = r["passed"]   # True | False | None (Error)
        # Explicitly passed before, explicitly failed now → newly failing
        if pp is True and rp is False:
            nf.append((p, r))
        # Explicitly failed before, now passing → newly passing
        elif pp is False and rp is True:
            np_.append((p, r))
        # Passed both runs, score dropped enough to flag
        elif pp is True and rp is True and p["score"] is not None and r["score"] is not None:
            if (p["score"] - r["score"]) >= delta:
                drops.append((p, r))
    return nf, np_, drops



# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(t: Any) -> str:
    return html.escape(str(t or ""), quote=True)


def _score_fmt(score: float | None) -> str:
    if score is None:
        return "<span class='badge badge-error'>None</span>"
    return f"{score:.2f}"


def _diff_table(pairs: list, color_cls: str, heading: str) -> str:
    if not pairs:
        return (
            f"<div class='section-card'><h3>{heading}</h3>"
            f"<p class='muted'>None &mdash; all good! &#x2705;</p></div>"
        )
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td>"
        f"<td>{_esc(r['metric'])}</td>"
        f"<td>{_score_fmt(p['score'])} &rarr; {_score_fmt(r['score'])}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for p, r in pairs
    )
    return (
        f"<div class='section-card {color_cls}'><h3>{heading}</h3>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Prev &rarr; Latest</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )


def _failing_table(latest_rows: list[dict]) -> str:
    failing = [r for r in latest_rows if r["passed"] is False]
    if not failing:
        return (
            "<div class='section-card'>"
            "<h3>&#x1F389; All checks passed in the latest run!</h3></div>"
        )
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td>"
        f"<td>{_esc(r['metric'])}</td>"
        f"<td>{_score_fmt(r['score'])}</td>"
        f"<td>{(str(round(r['threshold'], 2)) if r['threshold'] is not None else '-')}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for r in failing
    )
    return (
        f"<div class='section-card failing-card'>"
        f"<h3>&#x274C; Failing Cases &mdash; Latest Run ({len(failing)} total)</h3>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Score</th><th>Threshold</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )


def _error_table(latest_rows: list[dict]) -> str:
    errors = [r for r in latest_rows if r["passed"] is None]
    if not errors:
        return ""
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td>"
        f"<td>{_esc(r['metric'])}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for r in errors
    )
    return (
        f"<div class='section-card error-card'>"
        f"<h3>&#x26A0;&#xFE0F; Evaluation Errors &mdash; Latest Run ({len(errors)} total)</h3>"
        f"<p class='muted'>Rows where the LLM judge returned no score (API error / timeout). "
        f"These are not counted as failures.</p>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )


def _kpi_html(latest_rows: list[dict], run_label: str, runs_count: int) -> str:
    total   = len(latest_rows)
    passed  = sum(1 for r in latest_rows if r["passed"] is True)
    errors  = sum(1 for r in latest_rows if r["passed"] is None)
    metrics = len({r["metric"] for r in latest_rows})
    rate    = passed / total if total else 0
    color   = GREEN if rate >= 0.8 else (AMBER if rate >= 0.5 else RED)
    return (
        f"<div class='kpi-bar'>"
        f"<div class='kpi-card' style='border-top:3px solid {color}'>"
        f"<div class='kpi-value' style='color:{color}'>{rate:.0%}</div>"
        f"<div class='kpi-label'>Overall Pass Rate</div></div>"
        f"<div class='kpi-card'>"
        f"<div class='kpi-value'>{passed}<span class='kpi-sub'>/{total}</span></div>"
        f"<div class='kpi-label'>Checks Passed</div></div>"
        f"<div class='kpi-card' style='border-top:3px solid {GREY}'>"
        f"<div class='kpi-value' style='color:{GREY}'>{errors}</div>"
        f"<div class='kpi-label'>Evaluation Errors</div></div>"
        f"<div class='kpi-card'>"
        f"<div class='kpi-value'>{metrics}</div>"
        f"<div class='kpi-label'>Distinct Metrics</div></div>"
        f"<div class='kpi-card'>"
        f"<div class='kpi-value'>{runs_count}</div>"
        f"<div class='kpi-label'>Historical Runs</div></div>"
        f"<div class='kpi-card'>"
        f"<div class='kpi-value kpi-ts'>{_esc(run_label)}</div>"
        f"<div class='kpi-label'>Latest Run</div></div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_DARK = "#0f0f1a"
_CARD = "#1a1a2e"
_ACCENT = "#7c5cfc"

_CSS = f"""
:root{{
  --bg:{_DARK};--card-bg:{_CARD};--border:rgba(255,255,255,0.08);
  --accent:{_ACCENT};--text:#e0e0e0;--muted:#888;
  --pass:{GREEN};--fail:{RED};--error:{GREY};--amber:{AMBER};
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',system-ui,sans-serif;line-height:1.6}}
header{{background:linear-gradient(135deg,{_CARD} 0%,#16213e 50%,#0f3460 100%);
  padding:2.5rem 2rem 2rem;border-bottom:1px solid var(--border)}}
header h1{{font-size:2rem;font-weight:700;
  background:linear-gradient(90deg,{_ACCENT},#56ccf2);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
header p{{color:var(--muted);margin-top:.25rem;font-size:.9rem}}
main{{max-width:1400px;margin:0 auto;padding:2rem 1.5rem 4rem}}
.section-title{{font-size:1.3rem;font-weight:600;color:var(--text);
  margin:2.5rem 0 1rem;border-left:4px solid var(--accent);padding-left:.75rem}}
.kpi-bar{{display:flex;flex-wrap:wrap;gap:1rem;margin:1.5rem 0 2.5rem}}
.kpi-card{{flex:1;min-width:130px;background:var(--card-bg);border:1px solid var(--border);
  border-radius:12px;padding:1.2rem 1rem;text-align:center;
  border-top:3px solid var(--accent);transition:transform .2s,box-shadow .2s}}
.kpi-card:hover{{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.4)}}
.kpi-value{{font-size:2rem;font-weight:700;color:var(--accent);line-height:1.1}}
.kpi-value.kpi-ts{{font-size:1rem}}
.kpi-sub{{font-size:1rem;color:var(--muted)}}
.kpi-label{{font-size:.78rem;color:var(--muted);margin-top:.3rem;
  text-transform:uppercase;letter-spacing:.05em}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:1.5rem}}
.chart-card{{background:var(--card-bg);border:1px solid var(--border);
  border-radius:14px;padding:1rem;overflow:hidden}}
.section-card{{background:var(--card-bg);border:1px solid var(--border);
  border-radius:14px;padding:1.5rem;margin-bottom:1.5rem}}
.section-card.failing-card{{border-left:4px solid var(--fail)}}
.section-card.error-card{{border-left:4px solid var(--error)}}
.section-card.newly-failing{{border-left:4px solid var(--fail)}}
.section-card.newly-passing{{border-left:4px solid var(--pass)}}
.section-card.score-drop{{border-left:4px solid var(--amber)}}
.section-card h3{{font-size:1.05rem;font-weight:600;margin-bottom:1rem}}
.muted{{color:var(--muted);font-size:.9rem}}
.table-wrap{{overflow-x:auto}}
.data-table{{width:100%;border-collapse:collapse;font-size:.85rem}}
.data-table th,.data-table td{{padding:.6rem .8rem;text-align:left;
  border-bottom:1px solid var(--border)}}
.data-table th{{background:rgba(124,92,252,.12);color:var(--accent);
  font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}}
.data-table tr:hover td{{background:rgba(255,255,255,.03)}}
.reason-cell{{max-width:340px;white-space:pre-wrap;word-break:break-word;color:var(--muted)}}
.badge{{display:inline-block;border-radius:4px;padding:2px 8px;font-size:.75rem;font-weight:600}}
.badge-pass{{background:rgba(46,204,113,.2);color:var(--pass)}}
.badge-fail{{background:rgba(231,76,60,.2);color:var(--fail)}}
.badge-error{{background:rgba(149,165,166,.2);color:var(--error)}}
.diff-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem}}
.calibration-card{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:2rem}}
.calibration-title{{font-size:1.1rem;font-weight:600;margin-bottom:1rem;color:var(--text)}}
.calibration-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1rem}}
.calibration-stat{{background:rgba(255,255,255,0.03);padding:1rem;border-radius:8px;text-align:center}}
.calibration-val{{font-size:1.4rem;font-weight:700;margin-bottom:0.25rem}}
.calibration-lbl{{font-size:0.85rem;color:var(--muted)}}
footer{{text-align:center;color:var(--muted);padding:2rem;font-size:.82rem;
  border-top:1px solid var(--border)}}
@media(max-width:700px){{.chart-grid,.diff-grid{{grid-template-columns:1fr}}
  .kpi-bar{{flex-direction:column}}}}
"""

def _calibration_html(latest_rows: list[dict]) -> str:
    """Computes Cohen's Kappa for dual evaluations and builds the HTML section."""
    def _calc_kappa(rows: list[dict]):
        h_labels = []
        l_labels = []
        for r in rows:
            if r.get("human_score") is not None and r.get("passed") is not None:
                h_labels.append(1 if r["human_score"] >= 0.5 else 0)
                l_labels.append(1 if r["passed"] else 0)

        n = len(h_labels)
        if n == 0:
            return 0, 0.0, 0.0, "N/A", GREY

        a = sum(1 for h, l in zip(h_labels, l_labels) if h == 1 and l == 1)
        b = sum(1 for h, l in zip(h_labels, l_labels) if h == 1 and l == 0)
        c = sum(1 for h, l in zip(h_labels, l_labels) if h == 0 and l == 1)
        d = sum(1 for h, l in zip(h_labels, l_labels) if h == 0 and l == 0)

        p_o = (a + d) / n

        p_h1 = (a + b) / n
        p_h0 = (c + d) / n
        p_l1 = (a + c) / n
        p_l0 = (b + d) / n

        p_e = (p_h1 * p_l1) + (p_h0 * p_l0)
        if p_e == 1.0:
            kappa = 1.0 if p_o == 1.0 else 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        if kappa < 0:
            interp, color = "Poor", RED
        elif kappa <= 0.20:
            interp, color = "Slight", RED
        elif kappa <= 0.40:
            interp, color = "Fair", AMBER
        elif kappa <= 0.60:
            interp, color = "Moderate", AMBER
        elif kappa <= 0.80:
            interp, color = "Substantial (Calibrated)", GREEN
        else:
            interp, color = "Almost perfect (Highly Calibrated)", GREEN

        return n, p_o, kappa, interp, color

    n, p_o, kappa, interp, color = _calc_kappa(latest_rows)
    if n == 0:
        return ""

    success_msg = (
        f"<div style='color:{GREEN};font-weight:600;margin-top:1rem'>&#x2705; SUCCESS: Your LLM judge is calibrated overall.</div>"
        if kappa >= 0.60 else
        f"<div style='color:{RED};font-weight:600;margin-top:1rem'>&#x274C; WARNING: Your LLM judge is not well calibrated overall.</div>"
    )

    # Per-metric breakdown
    by_metric = defaultdict(list)
    for r in latest_rows:
        by_metric[r["metric"]].append(r)
        
    metric_rows_html = []
    for metric, m_rows in sorted(by_metric.items()):
        mn, mp_o, mkappa, minterp, mcolor = _calc_kappa(m_rows)
        if mn > 0:
            metric_rows_html.append(f"""
            <tr>
                <td style="font-weight:600">{html.escape(metric)}</td>
                <td>{mn}</td>
                <td>{mp_o*100:.1f}%</td>
                <td style="color:{mcolor};font-weight:600">{mkappa:.4f}</td>
                <td style="color:{mcolor}">{minterp}</td>
            </tr>
            """)
            
    breakdown_table = ""
    if metric_rows_html:
        breakdown_table = f"""
        <table class="data-table" style="width:100%;margin-top:2rem">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Cases</th>
                    <th>Agreement</th>
                    <th>Cohen's Kappa</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(metric_rows_html)}
            </tbody>
        </table>
        """

    return f"""
    <h2 class="section-title">&#x1F3AF; Judge Calibration Results</h2>
    <div class="calibration-card">
        <div class="calibration-grid">
            <div class="calibration-stat">
                <div class="calibration-val">{n}</div>
                <div class="calibration-lbl">Shared Cases Evaluated</div>
            </div>
            <div class="calibration-stat">
                <div class="calibration-val">{p_o*100:.2f}%</div>
                <div class="calibration-lbl">Percentage Agreement</div>
            </div>
            <div class="calibration-stat">
                <div class="calibration-val" style="color:{color}">{kappa:.4f}</div>
                <div class="calibration-lbl">Cohen's Kappa</div>
            </div>
        </div>
        <div style="color:{color};font-weight:600;text-align:center">Overall: {interp}</div>
        <div style="text-align:center">{success_msg}</div>
        {breakdown_table}
    </div>
    """

def chart_security_posture(latest_rows: list[dict]) -> str:
    adv_file = ROOT / "datasets" / "adversarial_testdata.json"
    if not adv_file.exists():
        return ""
    try:
        import json
        with adv_file.open("r", encoding="utf-8") as f:
            adv_data = json.load(f)
    except Exception:
        return ""
    
    attack_meta = {}
    for cat, cases in adv_data.items():
        if isinstance(cases, list):
            for c in cases:
                if "attack_id" in c:
                    attack_meta[c["attack_id"]] = {
                        "category": cat,
                        "vector": c.get("attack_vector", ""),
                        "severity": c.get("severity", "unknown").lower(),
                    }
    
    adv_rows = [r for r in latest_rows if r["case"] in attack_meta]
    if not adv_rows:
        return ""
    
    cat_counts = defaultdict(lambda: {"passed": 0, "total": 0})
    sev_counts = defaultdict(lambda: {"passed": 0, "total": 0})
    worst_offenders = []
    
    for r in adv_rows:
        meta = attack_meta[r["case"]]
        cat, sev = meta["category"], meta["severity"]
        cat_counts[cat]["total"] += 1
        sev_counts[sev]["total"] += 1
        if r["passed"] is True:
            cat_counts[cat]["passed"] += 1
            sev_counts[sev]["passed"] += 1
        elif r["passed"] is False:
            worst_offenders.append({
                "attack_id": r["case"], "attack_vector": meta["vector"],
                "metric": r["metric"], "score": r["score"], "reason": r["reason"]
            })
            
    cats = sorted(cat_counts.keys())
    cat_rates = [cat_counts[c]["passed"]/cat_counts[c]["total"] if cat_counts[c]["total"] else 0 for c in cats]
    fig_cat = go.Figure(go.Bar(
        x=cats, y=cat_rates, marker_color="#e67e22", text=[f"{r:.0%}" for r in cat_rates], textposition="auto"
    ))
    fig_cat.update_layout(
        title="Pass Rate per Attack Category",
        yaxis=dict(title="Pass Rate", tickformat=".0%", range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), height=350, margin=dict(l=40, r=20, t=50, b=50)
    )
    fig_cat.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    
    sevs = ["critical", "high", "medium", "low"]
    sevs_present = [s for s in sevs if sev_counts[s]["total"] > 0]
    sev_rates = [sev_counts[s]["passed"]/sev_counts[s]["total"] for s in sevs_present]
    fig_sev = go.Figure(go.Bar(
        x=sevs_present, y=sev_rates, marker_color="#e91e63", text=[f"{r:.0%}" for r in sev_rates], textposition="auto"
    ))
    fig_sev.update_layout(
        title="Pass Rate by Severity",
        yaxis=dict(title="Pass Rate", tickformat=".0%", range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), height=350, margin=dict(l=40, r=20, t=50, b=50)
    )
    fig_sev.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    
    c_cat_html = pio.to_html(fig_cat, full_html=False, include_plotlyjs=False)
    c_sev_html = pio.to_html(fig_sev, full_html=False, include_plotlyjs=False)
    
    worst_offenders.sort(key=lambda x: x["score"] if x["score"] is not None else 0)
    rows_html = "".join(
        f"<tr><td><code>{_esc(w['attack_id'])}</code></td><td>{_esc(w['attack_vector'])}</td><td>{_esc(w['metric'])}</td><td>{_score_fmt(w['score'])}</td><td class='reason-cell'>{_esc(w['reason'])}</td></tr>"
        for w in worst_offenders[:20]
    )
    table_html = (
        f"<div class='section-card failing-card'><h3>Worst Offenders (Failed Attacks)</h3>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Attack ID</th><th>Attack Vector</th><th>Metric</th><th>Score</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html if rows_html else '<tr><td colspan=5>No failed adversarial tests!</td></tr>'}</tbody></table></div></div>"
    )
    return f"""
    <h2 class="section-title">&#x1F6E1;&#xFE0F; Security Posture</h2>
    <div class="chart-grid">
        <div class="chart-card">{{c_cat_html}}</div>
        <div class="chart-card">{{c_sev_html}}</div>
    </div>
    {{table_html}}
    """

# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

def build_html(
    all_rows: list[dict],
    run_groups: list[tuple[str, list[dict]]],
    last_n: int | None,
    built_at: str,
) -> str:
    if not run_groups:
        return "<html><body><h1>No evaluation data found.</h1></body></html>"

    # Construct a "Latest State" containing the most recent run for *each* metric.
    # This ensures that piecemeal runs (e.g. running LLM at 10:00 and RAG at 11:00)
    # both show up in the snapshot charts.
    latest_stamps = {}
    for r in all_rows:
        m = r["metric"]
        if m not in latest_stamps or r["run_id"] > latest_stamps[m]:
            latest_stamps[m] = r["run_id"]
            
    latest_rows = [r for r in all_rows if r["run_id"] == latest_stamps[r["metric"]]]
    latest_label = "Latest (Combined)"

    c_pie = chart_overall_pie(latest_rows)
    c_sunburst = chart_category_sunburst(latest_rows)
    c1 = chart_pass_fail(latest_rows)
    c2 = chart_trend(run_groups, last_n)
    c3 = chart_category_trend(run_groups, last_n)
    c4 = chart_test_case_heatmap(run_groups, last_n)
    kpi = _kpi_html(latest_rows, latest_label, len(run_groups))
    calibration = _calibration_html(latest_rows)
    security_section = chart_security_posture(latest_rows)

    # Embed Plotly offline JS once using the official API
    plotly_script = f"<script type='text/javascript'>{get_plotlyjs()}</script>"

    # Construct the "Previous State" for the regression diff
    prev_stamps = {}
    for r in all_rows:
        m = r["metric"]
        if r["run_id"] < latest_stamps[m]:
            if m not in prev_stamps or r["run_id"] > prev_stamps[m]:
                prev_stamps[m] = r["run_id"]
                
    prev_rows = [r for r in all_rows if r["run_id"] == prev_stamps.get(r["metric"])]

    if prev_rows:
        nf, np_, drops = compute_diff(prev_rows, latest_rows, cfg.REGRESSION_DELTA)
        diff_section = (
            f"<h2 class='section-title'>&#x1F500; Run-over-Run Regression Diff"
            f"<span style='font-size:.8rem;font-weight:400;color:var(--muted)'>"
            f"&nbsp;Previous &rarr; {latest_label}</span></h2>"
            f"<div class='diff-grid'>"
            f"{_diff_table(nf,    'newly-failing', '&#x1F534; Newly Failing')}"
            f"{_diff_table(np_,   'newly-passing', '&#x1F7E2; Newly Passing')}"
            f"{_diff_table(drops, 'score-drop',    f'&#x1F7E1; Score Drops &ge; {cfg.REGRESSION_DELTA}')}"
            f"</div>"
        )
    else:
        diff_section = (
            "<h2 class='section-title'>&#x1F500; Regression Diff</h2>"
            "<div class='section-card'>"
            "<p class='muted'>Only one run found per metric &mdash; run the eval suite again to see a diff.</p>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenAI QA Eval &mdash; Dashboard</title>
<meta name="description" content="Aggregated evaluation dashboard for genai-qa-eval — pass rates, trends, regressions.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
{plotly_script}
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
  {_CSS}
  .export-btns {{ position: absolute; top: 1rem; right: 1rem; display: flex; gap: 0.5rem; }}
  .export-btn {{
      background: var(--surface); color: var(--text); border: 1px solid var(--border);
      padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600;
      cursor: pointer; transition: all 0.2s;
  }}
  .export-btn:hover {{ background: var(--border); color: #fff; }}
  @media print {{
      .export-btns {{ display: none !important; }}
      body {{ background: #fff; color: #000; }}
      .section-card, .chart-card, .calibration-card {{ border: 1px solid #ddd; background: #fff; }}
  }}
</style>
<script>
function downloadPNG() {{
    const btn = document.getElementById('png-btn');
    const oldText = btn.innerText;
    btn.innerText = 'Capturing...';
    
    // Brief timeout to allow text to update
    setTimeout(() => {{
        html2canvas(document.body, {{
            backgroundColor: '#0a0a0f',
            scale: 2 // higher res
        }}).then(canvas => {{
            const link = document.createElement('a');
            link.download = 'genai-qa-eval-dashboard.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            btn.innerText = oldText;
        }});
    }}, 100);
}}
</script>
</head>
<body>
<header style="position: relative;">
  <div class="export-btns">
      <button class="export-btn" onclick="window.print()">&#x1F4C4; Save as PDF</button>
      <button class="export-btn" id="png-btn" onclick="downloadPNG()">&#x1F4F7; Save as PNG</button>
  </div>
  <h1>&#x1F9EA; GenAI QA Eval &mdash; Dashboard</h1>
  <p>Offline evaluation reports viewer. Compare pass rates, run-over-run regressions, and test case stability over time.</p>
</header>
<main>
  {kpi}

  {calibration}

  <h2 class="section-title">&#x1F4CA; The Dashboard Snapshot</h2>
  <div class="chart-grid">
    <div class="chart-card">{c_pie}</div>
    <div class="chart-card">{c_sunburst}</div>
  </div>
  <div class="chart-grid">
    <div class="chart-card" style="grid-column: 1 / -1;">{c1}</div>
  </div>
  
  <h2 class="section-title">&#x1F4C8; Time-Wise Trends &amp; Stability</h2>
  <div class="chart-grid">
    <div class="chart-card">{c2}</div>
    <div class="chart-card">{c3}</div>
  </div>
  
  <div class="chart-card" style="margin-top: 1.5rem;">
    {c4}
  </div>

  {security_section}

  {diff_section}

  <h2 class="section-title">&#x1F6A8; Failing Cases &mdash; Latest Run</h2>
  {_failing_table(latest_rows)}
  {_error_table(latest_rows)}
</main>
<footer>genai-qa-eval dashboard &mdash; generated {_esc(built_at)} &mdash; offline-capable</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build an offline HTML dashboard from genai-qa-eval CSV reports."
    )
    p.add_argument(
        "--reports-dir", default=str(ROOT / "reports"),
        help="Root of the reports tree (default: <repo>/reports)",
    )
    p.add_argument(
        "--out", default=None,
        help="Output HTML path (default: <reports-dir>/dashboard.html)",
    )
    p.add_argument(
        "--last", type=int, default=None, metavar="N",
        help="Limit trend chart to the N most-recent runs",
    )
    p.add_argument(
        "--open", action="store_true",
        help="Open the dashboard in the default browser after building",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args        = _parse_args(argv)
    reports_dir = Path(args.reports_dir).resolve()
    out_path    = Path(args.out).resolve() if args.out else reports_dir / "dashboard.html"

    print(f"[build_dashboard] Scanning {reports_dir} ...")
    all_rows = load_all(reports_dir)

    if not all_rows:
        print(
            "[ERROR] No metric CSVs found. Run the eval suite first so that "
            "reports/ contains *_report_*.csv files.",
            file=sys.stderr,
        )
        return 1

    run_groups    = group_runs(all_rows, bucket_seconds=cfg.RUN_BUCKET_SECONDS)
    metrics_count = len({r["metric"] for r in all_rows})
    print(
        f"[build_dashboard] Loaded {len(all_rows)} rows across "
        f"{len(run_groups)} run(s) from {metrics_count} metric(s)."
    )

    built_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = build_html(all_rows, run_groups, last_n=args.last, built_at=built_at)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[build_dashboard] Dashboard written -> {out_path}")

    if args.open:
        if os.name == 'nt':
            os.startfile(str(out_path))
        else:
            webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
