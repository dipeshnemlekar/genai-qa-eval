import base64, pathlib

# The full source of build_dashboard.py encoded in base64 to avoid shell escaping issues
SRC = r"""from __future__ import annotations
import argparse, csv, html, re, sys, webbrowser
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
        CATEGORY_LABELS    = {"rag": "RAG", "safety": "Safety", "llm": "LLM Quality"}
        METRIC_CATEGORY    = {}
        LOWER_IS_BETTER    = {"Hallucination", "Pii Leakage", "Bias", "Toxicity"}
        THRESHOLDS         = {}
        REGRESSION_DELTA   = 0.1
        RUN_BUCKET_SECONDS = 120
    cfg = _Cfg()

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    print("[ERROR] plotly is required: pip install plotly", file=sys.stderr)
    sys.exit(1)

STAMP_RE  = re.compile(r"_report_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$")
METRIC_RE = re.compile(r"^(.+?)_report_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv$")

ALIASES = {
    "case":      {"tc no", "case", "name", "input", "test_case", "test_name", "test case"},
    "score":     {"judge score", "score", "metric_score", "value"},
    "threshold": {"threshold", "min_score"},
    "passed":    {"passed", "success", "result", "status"},
    "reason":    {"llm reason", "reason", "explanation", "judge_reason"},
}
REQUIRED_ALIASES = {"case", "score"}
GREEN  = "#2ecc71"; RED   = "#e74c3c"; GREY  = "#95a5a6"
AMBER  = "#f39c12"; BLUE  = "#3498db"
LINE_COLORS = [
    "#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
]

def _resolve_aliases(fieldnames):
    lower_map = {f.strip().lower(): f for f in (fieldnames or [])}
    result = {}
    for norm, candidates in ALIASES.items():
        hit = next((lower_map[c] for c in candidates if c in lower_map), None)
        result[norm] = hit
    missing = [k for k in REQUIRED_ALIASES if not result.get(k)]
    if missing:
        raise ValueError(f"Required column(s) {missing} not found. Headers: {list(fieldnames or [])}")
    return result

def _to_float(v):
    s = str(v).strip().lower() if v is not None else ""
    if s in ("", "none", "null", "na", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _derive_passed(score):
    """Infer pass/fail from judge score: 1.0 = pass, 0.0 = fail, None = Error."""
    if score is None:
        return None
    return score >= 1.0

def _to_bool_from_col(v):
    return str(v).strip().lower() in {"1", "true", "pass", "passed", "yes", "success"}

def _slug_to_name(slug):
    return slug.replace("_", " ").title()

def _category_display(folder, metric):
    label = cfg.CATEGORY_LABELS.get(folder.lower())
    if label:
        return label
    return cfg.METRIC_CATEGORY.get(metric, "Other")

def load_csv(path):
    nm = METRIC_RE.match(path.name)
    sm = STAMP_RE.search(path.name)
    if not nm or not sm:
        raise ValueError(f"Filename pattern mismatch: {path.name}")
    metric    = _slug_to_name(nm.group(1))
    run_stamp = datetime.strptime(sm.group(1), "%Y-%m-%d_%H-%M-%S")
    folder    = path.parent.name.lower()
    category  = _category_display(folder, metric)
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        a = _resolve_aliases(reader.fieldnames)
        for raw in reader:
            case_val = (raw.get(a["case"]) or "").strip()
            if not case_val:
                continue
            score     = _to_float(raw.get(a["score"])) if a.get("score") else None
            reason    = (raw.get(a["reason"]) or "").strip() if a.get("reason") else ""
            thr_raw   = raw.get(a["threshold"]) if a.get("threshold") else None
            threshold = _to_float(thr_raw) if thr_raw else cfg.THRESHOLDS.get(metric)
            passed_raw = raw.get(a["passed"]) if a.get("passed") else None
            if passed_raw and str(passed_raw).strip():
                passed = _to_bool_from_col(passed_raw)
            else:
                passed = _derive_passed(score)
            rows.append({
                "case":      case_val[:120],
                "metric":    metric,
                "category":  category,
                "folder":    folder,
                "score":     score,
                "threshold": threshold,
                "passed":    passed,
                "reason":    reason,
                "run_id":    run_stamp,
            })
    return rows

def load_all(reports_dir):
    all_rows = []
    for csv_path in sorted(reports_dir.rglob("*.csv")):
        if "_report_" not in csv_path.name:
            continue
        try:
            all_rows.extend(load_csv(csv_path))
        except Exception as exc:
            print(f"[warn] Skipping {csv_path.name}: {exc}")
    return all_rows

def group_runs(rows, bucket_seconds=120):
    all_stamps = sorted({r["run_id"] for r in rows})
    if not all_stamps:
        return []
    buckets, current = [], [all_stamps[0]]
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

def _state(row):
    if row["passed"] is None:
        return "Error"
    return "Passed" if row["passed"] else "Failed"

# ── Charts ──────────────────────────────────────────────────────────────────

def chart_pass_fail(latest_rows):
    counts = defaultdict(lambda: {"Passed": 0, "Failed": 0, "Error": 0})
    for row in latest_rows:
        counts[row["metric"]][_state(row)] += 1
    metrics = sorted(counts.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Passed", x=metrics,
        y=[counts[m]["Passed"] for m in metrics], marker_color=GREEN,
        text=[counts[m]["Passed"] for m in metrics], textposition="inside"))
    fig.add_trace(go.Bar(name="Failed", x=metrics,
        y=[counts[m]["Failed"] for m in metrics], marker_color=RED,
        text=[counts[m]["Failed"] for m in metrics], textposition="inside"))
    fig.add_trace(go.Bar(name="Error",  x=metrics,
        y=[counts[m]["Error"] for m in metrics],  marker_color=GREY,
        text=[counts[m]["Error"] for m in metrics],  textposition="inside"))
    fig.update_layout(barmode="stack",
        title=dict(text="Per-Metric Pass / Fail (Latest Run)", font=dict(size=18)),
        xaxis_title="Metric", yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=-0.3),
        margin=dict(l=50, r=20, t=60, b=130), height=420)
    fig.update_xaxes(tickangle=-30, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def chart_trend(run_groups, last_n):
    groups = run_groups[-last_n:] if last_n else run_groups
    all_metrics = sorted({r["metric"] for _, rw in groups for r in rw})
    labels = [lb for lb, _ in groups]
    fig = go.Figure()
    for idx, metric in enumerate(all_metrics):
        rates = []
        for _, rrows in groups:
            m = [r for r in rrows if r["metric"] == metric]
            rates.append(sum(1 for r in m if r["passed"] is True) / len(m) if m else None)
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        fig.add_trace(go.Scatter(x=labels, y=rates, name=metric, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7), connectgaps=False))
    overall = [
        sum(1 for r in rrows if r["passed"] is True) / len(rrows) if rrows else None
        for _, rrows in groups
    ]
    fig.add_trace(go.Scatter(x=labels, y=overall, name="Overall", mode="lines+markers",
        line=dict(color="#ffffff", width=3, dash="dash"),
        marker=dict(size=9, color="#ffffff"), connectgaps=False))
    fig.update_layout(title=dict(text="Pass-Rate Trend Across Runs", font=dict(size=18)),
        xaxis_title="Run",
        yaxis=dict(title="Pass Rate", tickformat=".0%", range=[0, 1.05],
                   gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=-0.35),
        margin=dict(l=60, r=20, t=60, b=150), height=440)
    fig.update_xaxes(tickangle=-30, gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def chart_score_dist(latest_rows):
    m_scores   = defaultdict(list)
    thresholds = {}
    for row in latest_rows:
        if row["score"] is not None:
            m_scores[row["metric"]].append(row["score"])
            thresholds.setdefault(row["metric"], row["threshold"])
    if not m_scores:
        fig = go.Figure()
        fig.update_layout(title="Score Distribution (no numeric scores in latest run)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), height=350)
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)
    metrics = sorted(m_scores.keys())
    fig = go.Figure()
    for idx, metric in enumerate(metrics):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        fig.add_trace(go.Box(y=m_scores[metric], name=metric, boxpoints="all",
            jitter=0.3, pointpos=-1.5,
            marker=dict(color=color, size=6, opacity=0.7), line=dict(color=color)))
        thr = thresholds.get(metric)
        if thr is not None:
            is_lower = metric in cfg.LOWER_IS_BETTER
            sym = "<=" if is_lower else ">="
            fig.add_shape(type="line", x0=idx - 0.45, x1=idx + 0.45, y0=thr, y1=thr,
                line=dict(color="white", dash="dot", width=1.5))
            fig.add_annotation(x=metric, y=thr,
                text=f"thr({sym}{thr})", showarrow=False,
                font=dict(size=9, color="white"), yshift=8)
    fig.update_layout(title=dict(text="Score Distribution vs Threshold (Latest Run)", font=dict(size=18)),
        yaxis_title="Score", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), showlegend=False,
        margin=dict(l=60, r=20, t=60, b=100), height=420)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def chart_category_donut(latest_rows):
    cat_pass  = defaultdict(int)
    cat_total = defaultdict(int)
    for row in latest_rows:
        cat_total[row["category"]] += 1
        if row["passed"] is True:
            cat_pass[row["category"]] += 1
    cats   = sorted(cat_total.keys())
    labels = [f"{c}<br>{cat_pass[c]}/{cat_total[c]}" for c in cats]
    values = [cat_total[c] for c in cats]
    rates  = [round(cat_pass[c] / cat_total[c], 4) if cat_total[c] else 0 for c in cats]
    colors = [LINE_COLORS[i % len(LINE_COLORS)] for i in range(len(cats))]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent", customdata=rates,
        hovertemplate="<b>%{label}</b><br>Pass Rate: %{customdata:.0%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Category Rollup - Pass Rate (Latest Run)", font=dict(size=18)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=20, r=20, t=60, b=20), height=400, showlegend=True,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

# ── HTML helpers ─────────────────────────────────────────────────────────────

def _esc(t):
    return html.escape(str(t or ""), quote=True)

def _score_fmt(score):
    if score is None:
        return "<span class='badge badge-error'>None</span>"
    return f"{score:.2f}"

def compute_diff(prev_rows, latest_rows, delta=0.1):
    key      = lambda r: (r["case"], r["metric"])
    prev_map = {key(r): r for r in prev_rows}
    nf, np_, drops = [], [], []
    for r in latest_rows:
        p = prev_map.get(key(r))
        if not p:
            continue
        pp = p["passed"] if p["passed"] is not None else False
        rp = r["passed"] if r["passed"] is not None else False
        if pp and not rp:
            nf.append((p, r))
        elif not pp and rp:
            np_.append((p, r))
        elif pp and rp and p["score"] is not None and r["score"] is not None:
            if (p["score"] - r["score"]) >= delta:
                drops.append((p, r))
    return nf, np_, drops

def _diff_table(pairs, color_cls, heading):
    if not pairs:
        return (
            f"<div class='section-card'><h3>{_esc(heading)}</h3>"
            f"<p class='muted'>None - all good!</p></div>"
        )
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td>"
        f"<td>{_esc(r['metric'])}</td>"
        f"<td>{_score_fmt(p['score'])} &rarr; {_score_fmt(r['score'])}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for p, r in pairs
    )
    return (
        f"<div class='section-card {color_cls}'><h3>{_esc(heading)}</h3>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Prev &rarr; Latest</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )

def _failing_table(latest_rows):
    failing = [r for r in latest_rows if r["passed"] is False]
    if not failing:
        return "<div class='section-card'><h3>All checks passed in the latest run! &#x2705;</h3></div>"
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td><td>{_esc(r['metric'])}</td>"
        f"<td>{_score_fmt(r['score'])}</td>"
        f"<td>{f\"{r['threshold']:.2f}\" if r['threshold'] is not None else '-'}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for r in failing
    )
    return (
        f"<div class='section-card failing-card'>"
        f"<h3>&#x274C; Failing Cases - Latest Run ({len(failing)} total)</h3>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Score</th><th>Threshold</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )

def _error_table(latest_rows):
    errors = [r for r in latest_rows if r["passed"] is None]
    if not errors:
        return ""
    rows_html = "".join(
        f"<tr><td><code>{_esc(r['case'])}</code></td><td>{_esc(r['metric'])}</td>"
        f"<td class='reason-cell'>{_esc(r['reason'])}</td></tr>"
        for r in errors
    )
    return (
        f"<div class='section-card error-card'>"
        f"<h3>&#x26A0;&#xFE0F; Evaluation Errors - Latest Run ({len(errors)} total)</h3>"
        f"<p class='muted'>Rows where the LLM judge returned no score (API error / timeout).</p>"
        f"<div class='table-wrap'><table class='data-table'><thead><tr>"
        f"<th>Test Case</th><th>Metric</th><th>Reason</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div></div>"
    )

def _kpi_html(latest_rows, run_label, runs_count):
    total   = len(latest_rows)
    passed  = sum(1 for r in latest_rows if r["passed"] is True)
    errors  = sum(1 for r in latest_rows if r["passed"] is None)
    metrics = len({r["metric"] for r in latest_rows})
    rate    = passed / total if total else 0
    color   = GREEN if rate >= 0.8 else (AMBER if rate >= 0.5 else RED)
    return f"""
    <div class='kpi-bar'>
      <div class='kpi-card' style='border-top:3px solid {color}'>
        <div class='kpi-value' style='color:{color}'>{rate:.0%}</div>
        <div class='kpi-label'>Overall Pass Rate</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-value'>{passed}<span class='kpi-sub'>/{total}</span></div>
        <div class='kpi-label'>Checks Passed</div>
      </div>
      <div class='kpi-card' style='border-top:3px solid {GREY}'>
        <div class='kpi-value' style='color:{GREY}'>{errors}</div>
        <div class='kpi-label'>Evaluation Errors</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-value'>{metrics}</div>
        <div class='kpi-label'>Distinct Metrics</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-value'>{runs_count}</div>
        <div class='kpi-label'>Historical Runs</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-value kpi-ts'>{_esc(run_label)}</div>
        <div class='kpi-label'>Latest Run</div>
      </div>
    </div>"""

CSS = """:root{--bg:#0f0f1a;--card-bg:#1a1a2e;--border:rgba(255,255,255,0.08);--accent:#7c5cfc;--text:#e0e0e0;--muted:#888;--pass:#2ecc71;--fail:#e74c3c;--error:#95a5a6;--amber:#f39c12}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',system-ui,sans-serif;line-height:1.6}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:2.5rem 2rem 2rem;border-bottom:1px solid var(--border)}
header h1{font-size:2rem;font-weight:700;background:linear-gradient(90deg,#7c5cfc,#56ccf2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
header p{color:var(--muted);margin-top:.25rem;font-size:.9rem}
main{max-width:1400px;margin:0 auto;padding:2rem 1.5rem 4rem}
.section-title{font-size:1.3rem;font-weight:600;color:var(--text);margin:2.5rem 0 1rem;border-left:4px solid var(--accent);padding-left:.75rem}
.kpi-bar{display:flex;flex-wrap:wrap;gap:1rem;margin:1.5rem 0 2.5rem}
.kpi-card{flex:1;min-width:130px;background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1rem;text-align:center;border-top:3px solid var(--accent);transition:transform .2s,box-shadow .2s}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.4)}
.kpi-value{font-size:2rem;font-weight:700;color:var(--accent);line-height:1.1}
.kpi-value.kpi-ts{font-size:1rem}
.kpi-sub{font-size:1rem;color:var(--muted)}
.kpi-label{font-size:.78rem;color:var(--muted);margin-top:.3rem;text-transform:uppercase;letter-spacing:.05em}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:1.5rem}
.chart-card{background:var(--card-bg);border:1px solid var(--border);border-radius:14px;padding:1rem;overflow:hidden}
.section-card{background:var(--card-bg);border:1px solid var(--border);border-radius:14px;padding:1.5rem;margin-bottom:1.5rem}
.section-card.failing-card{border-left:4px solid var(--fail)}.section-card.error-card{border-left:4px solid var(--error)}
.section-card.newly-failing{border-left:4px solid var(--fail)}.section-card.newly-passing{border-left:4px solid var(--pass)}
.section-card.score-drop{border-left:4px solid var(--amber)}
.section-card h3{font-size:1.05rem;font-weight:600;margin-bottom:1rem}
.muted{color:var(--muted);font-size:.9rem}
.table-wrap{overflow-x:auto}
.data-table{width:100%;border-collapse:collapse;font-size:.85rem}
.data-table th,.data-table td{padding:.6rem .8rem;text-align:left;border-bottom:1px solid var(--border)}
.data-table th{background:rgba(124,92,252,.12);color:var(--accent);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.data-table tr:hover td{background:rgba(255,255,255,.03)}
.reason-cell{max-width:340px;white-space:pre-wrap;word-break:break-word;color:var(--muted)}
.badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:.75rem;font-weight:600}
.badge-pass{background:rgba(46,204,113,.2);color:var(--pass)}.badge-fail{background:rgba(231,76,60,.2);color:var(--fail)}
.badge-error{background:rgba(149,165,166,.2);color:var(--error)}
.diff-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem}
footer{text-align:center;color:var(--muted);padding:2rem;font-size:.82rem;border-top:1px solid var(--border)}
@media(max-width:700px){.chart-grid,.diff-grid{grid-template-columns:1fr}.kpi-bar{flex-direction:column}}"""

def build_html(all_rows, run_groups, last_n, built_at):
    if not run_groups:
        return "<html><body><h1>No evaluation data found.</h1></body></html>"
    latest_label, latest_rows = run_groups[-1]
    c1 = chart_pass_fail(latest_rows)
    c2 = chart_trend(run_groups, last_n)
    c3 = chart_score_dist(latest_rows)
    c4 = chart_category_donut(latest_rows)
    kpi = _kpi_html(latest_rows, latest_label, len(run_groups))

    # Embed Plotly offline JS once
    tmp_html = pio.to_html(go.Figure(), full_html=True, include_plotlyjs=True)
    script_match = re.search(r"(<script[^>]*>[\s\S]*?</script>)", tmp_html)
    plotly_script = script_match.group(0) if script_match else ""

    if len(run_groups) >= 2:
        prev_label, prev_rows = run_groups[-2]
        nf, np_, drops = compute_diff(prev_rows, latest_rows, cfg.REGRESSION_DELTA)
        diff_section = (
            f"<h2 class='section-title'>&#x1F500; Run-over-Run Regression Diff "
            f"<span style='font-size:.8rem;font-weight:400;color:var(--muted)'>"
            f"&nbsp;{_esc(prev_label)} &rarr; {_esc(latest_label)}</span></h2>"
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
            "<p class='muted'>Only one run found &mdash; run the eval suite again to see a diff.</p>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenAI QA Eval &mdash; Dashboard</title>
<meta name="description" content="Aggregated evaluation dashboard for genai-qa-eval.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
{plotly_script}
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>&#x1F9EA; GenAI QA Eval &mdash; Dashboard</h1>
  <p>Built {_esc(built_at)} &middot; {len(run_groups)} run(s) &middot; {len(all_rows)} total checks</p>
</header>
<main>
  {kpi}
  <h2 class="section-title">&#x1F4CA; Per-Metric Pass / Fail &amp; Trend</h2>
  <div class="chart-grid">
    <div class="chart-card">{c1}</div>
    <div class="chart-card">{c2}</div>
  </div>
  <h2 class="section-title">&#x1F4C8; Score Distribution &amp; Category Breakdown</h2>
  <div class="chart-grid">
    <div class="chart-card">{c3}</div>
    <div class="chart-card">{c4}</div>
  </div>
  {diff_section}
  <h2 class="section-title">&#x1F6A8; Failing Cases &mdash; Latest Run</h2>
  {_failing_table(latest_rows)}
  {_error_table(latest_rows)}
</main>
<footer>genai-qa-eval dashboard &mdash; generated {_esc(built_at)} &mdash; offline-capable</footer>
</body>
</html>"""

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build an offline HTML dashboard from genai-qa-eval CSV reports.")
    p.add_argument("--reports-dir", default=str(ROOT / "reports"),
                   help="Root of the reports tree (default: <repo>/reports)")
    p.add_argument("--out", default=None,
                   help="Output HTML path (default: <reports-dir>/dashboard.html)")
    p.add_argument("--last", type=int, default=None, metavar="N",
                   help="Limit trend chart to the N most-recent runs")
    p.add_argument("--open", action="store_true",
                   help="Open the dashboard in the default browser after building")
    return p.parse_args(argv)

def main(argv=None):
    args        = _parse_args(argv)
    reports_dir = Path(args.reports_dir).resolve()
    out_path    = Path(args.out).resolve() if args.out else reports_dir / "dashboard.html"
    print(f"[build_dashboard] Scanning {reports_dir} ...")
    all_rows = load_all(reports_dir)
    if not all_rows:
        print(
            "[ERROR] No metric CSVs found. Run the eval suite first so that "
            "reports/ contains *_report_*.csv files.", file=sys.stderr)
        return 1
    run_groups    = group_runs(all_rows, bucket_seconds=cfg.RUN_BUCKET_SECONDS)
    metrics_count = len({r["metric"] for r in all_rows})
    print(
        f"[build_dashboard] Loaded {len(all_rows)} rows across "
        f"{len(run_groups)} run(s) from {metrics_count} metric(s).")
    built_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = build_html(all_rows, run_groups, last_n=args.last, built_at=built_at)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[build_dashboard] Dashboard written -> {out_path}")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

dst = pathlib.Path(r"c:\Users\DIPESH\PycharmProjects\genai-qa-eval\runners\build_dashboard.py")
dst.write_text(SRC, encoding="utf-8")
print(f"Written {len(SRC)} chars to {dst}")
