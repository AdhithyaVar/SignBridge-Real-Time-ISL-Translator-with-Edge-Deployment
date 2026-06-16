"""
reporter.py
-----------
Generates evaluation reports for the SignBridge ISL classifier.

Outputs (all saved to PROJECT/reports/)
----------------------------------------
confusion_matrix.png     — heatmap of (C, C) confusion matrix
per_class_f1.png         — horizontal bar chart of per-class F1 scores
evaluation_report.html   — full self-contained HTML report
evaluation_summary.json  — machine-readable metrics dict
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.evaluation.evaluator import EvaluationResult
from src.utils.class_labels import CLASS_LABELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
_PALETTE   = "Blues"
_FIG_DPI   = 120
_FONT_SIZE = 9


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    result:   EvaluationResult,
    save_dir: Path,
    normalise: bool = True,
) -> Path:
    """
    Save a confusion-matrix heatmap as confusion_matrix.png.

    Parameters
    ----------
    result    : EvaluationResult
    save_dir  : directory to save the PNG
    normalise : if True, show recall (row-normalised %) instead of raw counts

    Returns
    -------
    Path to saved PNG file.
    """
    cm    = result.cm.astype(float)
    names = result.class_names
    C     = len(names)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot  = cm / row_sums
        fmt      = ".2f"
        title    = f"Confusion Matrix (Recall %) — {result.split.upper()} set"
        vmax     = 1.0
    else:
        cm_plot = cm
        fmt     = "d"
        title   = f"Confusion Matrix (counts) — {result.split.upper()} set"
        vmax    = None

    fig, ax = plt.subplots(figsize=(max(10, C * 0.55), max(8, C * 0.50)))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=_PALETTE,
        xticklabels=names,
        yticklabels=names,
        linewidths=0.3,
        linecolor="#cccccc",
        vmin=0.0,
        vmax=vmax,
        annot_kws={"size": _FONT_SIZE},
        ax=ax,
    )

    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class",      fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=_FONT_SIZE)
    ax.tick_params(axis="y", rotation=0,  labelsize=_FONT_SIZE)

    # Highlight diagonal (correct predictions) with a border
    for i in range(C):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=False, edgecolor="green", lw=1.5
        ))

    plt.tight_layout()
    save_path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# Per-class F1 bar chart
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    result:   EvaluationResult,
    save_dir: Path,
) -> Path:
    """
    Save a horizontal bar chart of per-class F1 scores as per_class_f1.png.
    Classes are sorted by F1 score (worst → best) for easy identification
    of underperforming classes.

    Returns
    -------
    Path to saved PNG file.
    """
    per_class  = sorted(result.per_class, key=lambda x: x["f1"])
    names      = [d["class_name"] for d in per_class]
    f1_scores  = [d["f1"]         for d in per_class]
    precisions = [d["precision"]  for d in per_class]
    recalls    = [d["recall"]     for d in per_class]

    C   = len(names)
    fig, ax = plt.subplots(figsize=(10, max(6, C * 0.42)))

    # Colour bars by performance tier
    colors = [
        "#d32f2f" if f < 0.70 else
        "#f57c00" if f < 0.85 else
        "#388e3c"
        for f in f1_scores
    ]

    y_pos = np.arange(C)
    bars  = ax.barh(y_pos, f1_scores, color=colors,
                    height=0.6, edgecolor="white", linewidth=0.5)

    # Overlay precision and recall as dots
    ax.scatter(precisions, y_pos, marker="^", color="#1565C0",
               s=40, zorder=5, label="Precision", alpha=0.85)
    ax.scatter(recalls,    y_pos, marker="v", color="#6A1B9A",
               s=40, zorder=5, label="Recall",    alpha=0.85)

    # Value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax.text(
            min(val + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8, color="#333333",
        )

    # Reference lines
    ax.axvline(0.90, color="green",  linestyle="--", linewidth=1.2,
               label="Target (0.90)", alpha=0.7)
    ax.axvline(0.80, color="orange", linestyle=":",  linewidth=1.0,
               alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=_FONT_SIZE)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(
        f"Per-Class F1 / Precision / Recall — {result.split.upper()} set\n"
        f"(sorted worst → best F1)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_path = Path(save_dir) / "per_class_f1.png"
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved per-class F1 chart → {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def save_json_summary(
    test_result: EvaluationResult,
    val_result:  EvaluationResult,
    save_dir:    Path,
) -> Path:
    """Save a machine-readable JSON summary of all evaluation metrics."""

    def _result_to_dict(r: EvaluationResult) -> dict:
        return {
            "split":           r.split,
            "accuracy":        round(r.accuracy,        4),
            "top3_accuracy":   round(r.top3_accuracy,   4),
            "macro_f1":        round(r.macro_f1,        4),
            "macro_precision": round(r.macro_precision, 4),
            "macro_recall":    round(r.macro_recall,    4),
            "weighted_f1":     round(r.weighted_f1,     4),
            "num_samples":     r.num_samples,
            "target_met":      r.target_met,
            "per_class": [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in cls.items()}
                for cls in r.per_class
            ],
            "top10_confused_pairs": r.confused_pairs,
        }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "test":         _result_to_dict(test_result),
        "val":          _result_to_dict(val_result),
    }

    save_path = Path(save_dir) / "evaluation_summary.json"
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Saved JSON summary → {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    test_result:   EvaluationResult,
    val_result:    EvaluationResult,
    cm_path:       Path,
    f1_path:       Path,
    save_dir:      Path,
) -> Path:
    """
    Generate a self-contained HTML evaluation report.

    Parameters
    ----------
    test_result, val_result : EvaluationResult objects
    cm_path  : path to confusion_matrix.png
    f1_path  : path to per_class_f1.png
    save_dir : output directory

    Returns
    -------
    Path to evaluation_report.html
    """
    import base64

    def _img_b64(path: Path) -> str:
        """Embed image as base64 so HTML is fully self-contained."""
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _badge(value: float, threshold: float = 0.90) -> str:
        color = "#2e7d32" if value >= threshold else (
                "#f57c00" if value >= 0.75 else "#c62828")
        return (f'<span style="background:{color};color:#fff;'
                f'padding:2px 8px;border-radius:4px;font-weight:bold;">'
                f'{value*100:.2f}%</span>')

    def _per_class_table(result: EvaluationResult) -> str:
        rows = sorted(result.per_class, key=lambda x: x["f1"], reverse=True)
        html = """
        <table class="metrics-table">
          <thead>
            <tr>
              <th>#</th><th>Class</th>
              <th>Precision</th><th>Recall</th><th>F1</th>
              <th>Correct</th><th>Total</th>
            </tr>
          </thead>
          <tbody>
        """
        for i, row in enumerate(rows):
            f1_color = (
                "#2e7d32" if row["f1"] >= 0.90 else
                "#f57c00" if row["f1"] >= 0.75 else
                "#c62828"
            )
            html += f"""
            <tr>
              <td>{i+1}</td>
              <td><strong>{row['class_name']}</strong></td>
              <td>{row['precision']:.4f}</td>
              <td>{row['recall']:.4f}</td>
              <td style="color:{f1_color};font-weight:bold">{row['f1']:.4f}</td>
              <td>{row['correct']}</td>
              <td>{row['support']}</td>
            </tr>"""
        html += "</tbody></table>"
        return html

    def _confused_pairs_table(result: EvaluationResult) -> str:
        if not result.confused_pairs:
            return "<p>No confusion pairs found.</p>"
        html = """
        <table class="metrics-table">
          <thead>
            <tr><th>True Class</th><th>Predicted As</th><th>Count</th></tr>
          </thead><tbody>
        """
        for pair in result.confused_pairs[:10]:
            html += f"""
            <tr>
              <td>{pair['true_class']}</td>
              <td>{pair['pred_class']}</td>
              <td>{pair['count']}</td>
            </tr>"""
        html += "</tbody></table>"
        return html

    cm_b64 = _img_b64(cm_path) if cm_path.exists() else ""
    f1_b64 = _img_b64(f1_path) if f1_path.exists() else ""
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SignBridge — Evaluation Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif;
           background: #f4f6f9; color: #212121; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 26px; font-weight: 700; color: #1a237e;
          border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
    h2 {{ font-size: 18px; color: #283593; margin: 24px 0 10px; }}
    h3 {{ font-size: 14px; color: #455a64; margin: 16px 0 6px; }}
    .meta {{ color: #607d8b; font-size: 13px; margin-top: 4px; }}
    .card {{ background: #fff; border-radius: 8px;
             box-shadow: 0 1px 4px rgba(0,0,0,.12);
             padding: 20px; margin-bottom: 20px; }}
    .metric-grid {{ display: grid;
                   grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
                   gap: 12px; margin-bottom: 16px; }}
    .metric-box {{ background: #e8eaf6; border-radius: 6px;
                  padding: 14px 12px; text-align: center; }}
    .metric-box .val {{ font-size: 24px; font-weight: 700;
                        color: #1a237e; }}
    .metric-box .lbl {{ font-size: 11px; color: #546e7a;
                        margin-top: 2px; }}
    .metrics-table {{ width: 100%; border-collapse: collapse;
                      font-size: 13px; }}
    .metrics-table th {{ background: #1a237e; color: #fff;
                         padding: 8px 10px; text-align: left; }}
    .metrics-table td {{ padding: 6px 10px;
                         border-bottom: 1px solid #e0e0e0; }}
    .metrics-table tr:nth-child(even) td {{ background: #f5f5f5; }}
    .plot-img {{ max-width: 100%; border-radius: 6px;
                 box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
    .target-met   {{ color: #2e7d32; font-weight: bold; }}
    .target-unmet {{ color: #c62828; font-weight: bold; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media(max-width:700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<div class="container">

  <h1>SignBridge — ISL Evaluation Report</h1>
  <p class="meta">Generated: {ts} &nbsp;|&nbsp;
     Test samples: {test_result.num_samples} &nbsp;|&nbsp;
     Val samples: {val_result.num_samples} &nbsp;|&nbsp;
     Classes: {len(test_result.class_names)}
  </p>

  <!-- ── Summary metrics ─────────────────────────── -->
  <div class="card">
    <h2>Overall Performance</h2>
    <div class="two-col">
      <div>
        <h3>Test Set</h3>
        <div class="metric-grid">
          <div class="metric-box">
            <div class="val">{test_result.accuracy*100:.2f}%</div>
            <div class="lbl">Top-1 Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="val">{test_result.top3_accuracy*100:.2f}%</div>
            <div class="lbl">Top-3 Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="val">{test_result.macro_f1*100:.2f}%</div>
            <div class="lbl">Macro F1</div>
          </div>
          <div class="metric-box">
            <div class="val">{test_result.macro_precision*100:.2f}%</div>
            <div class="lbl">Macro Precision</div>
          </div>
          <div class="metric-box">
            <div class="val">{test_result.macro_recall*100:.2f}%</div>
            <div class="lbl">Macro Recall</div>
          </div>
          <div class="metric-box">
            <div class="val">{test_result.weighted_f1*100:.2f}%</div>
            <div class="lbl">Weighted F1</div>
          </div>
        </div>
        <p>Target (&gt;90%): <span class="{'target-met' if test_result.target_met else 'target-unmet'}">
          {'✓  ACHIEVED' if test_result.target_met else '✗  NOT YET'}</span>
        </p>
      </div>
      <div>
        <h3>Validation Set</h3>
        <div class="metric-grid">
          <div class="metric-box">
            <div class="val">{val_result.accuracy*100:.2f}%</div>
            <div class="lbl">Top-1 Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="val">{val_result.top3_accuracy*100:.2f}%</div>
            <div class="lbl">Top-3 Accuracy</div>
          </div>
          <div class="metric-box">
            <div class="val">{val_result.macro_f1*100:.2f}%</div>
            <div class="lbl">Macro F1</div>
          </div>
          <div class="metric-box">
            <div class="val">{val_result.macro_precision*100:.2f}%</div>
            <div class="lbl">Macro Precision</div>
          </div>
          <div class="metric-box">
            <div class="val">{val_result.macro_recall*100:.2f}%</div>
            <div class="lbl">Macro Recall</div>
          </div>
          <div class="metric-box">
            <div class="val">{val_result.weighted_f1*100:.2f}%</div>
            <div class="lbl">Weighted F1</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Confusion matrix ────────────────────────── -->
  <div class="card">
    <h2>Confusion Matrix (Test Set)</h2>
    {"<img class='plot-img' src='data:image/png;base64," + cm_b64 + "' alt='Confusion Matrix'>" if cm_b64 else "<p>Image not found.</p>"}
  </div>

  <!-- ── Per-class F1 chart ──────────────────────── -->
  <div class="card">
    <h2>Per-Class Performance (Test Set)</h2>
    {"<img class='plot-img' src='data:image/png;base64," + f1_b64 + "' alt='Per-class F1'>" if f1_b64 else "<p>Image not found.</p>"}
  </div>

  <!-- ── Per-class table ─────────────────────────── -->
  <div class="card">
    <h2>Per-Class Metrics Table (Test Set)</h2>
    {_per_class_table(test_result)}
  </div>

  <!-- ── Confused pairs ──────────────────────────── -->
  <div class="card">
    <div class="two-col">
      <div>
        <h2>Top Confused Pairs — Test</h2>
        {_confused_pairs_table(test_result)}
      </div>
      <div>
        <h2>Top Confused Pairs — Val</h2>
        {_confused_pairs_table(val_result)}
      </div>
    </div>
  </div>

  <!-- ── Worst classes ───────────────────────────── -->
  <div class="card">
    <div class="two-col">
      <div>
        <h2>5 Worst Classes (Test F1)</h2>
        <table class="metrics-table">
          <thead><tr><th>Class</th><th>F1</th><th>Correct/Total</th></tr></thead>
          <tbody>
          {"".join(f"<tr><td>{c['class_name']}</td><td style='color:#c62828'>{c['f1']:.4f}</td><td>{c['correct']}/{c['support']}</td></tr>" for c in test_result.worst_classes(5))}
          </tbody>
        </table>
      </div>
      <div>
        <h2>5 Best Classes (Test F1)</h2>
        <table class="metrics-table">
          <thead><tr><th>Class</th><th>F1</th><th>Correct/Total</th></tr></thead>
          <tbody>
          {"".join(f"<tr><td>{c['class_name']}</td><td style='color:#2e7d32'>{c['f1']:.4f}</td><td>{c['correct']}/{c['support']}</td></tr>" for c in test_result.best_classes(5))}
          </tbody>
        </table>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""

    save_path = Path(save_dir) / "evaluation_report.html"
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info(f"Saved HTML report → {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# Master reporter
# ---------------------------------------------------------------------------

def generate_all_reports(
    test_result: EvaluationResult,
    val_result:  EvaluationResult,
    reports_dir: Path,
) -> dict[str, Path]:
    """
    Generate all report artefacts and save them to reports_dir.

    Parameters
    ----------
    test_result, val_result : EvaluationResult
    reports_dir             : destination directory

    Returns
    -------
    dict mapping artefact name → Path
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating reports → {reports_dir}")

    paths = {}

    # 1. Confusion matrix (test)
    paths["confusion_matrix"] = plot_confusion_matrix(
        test_result, reports_dir, normalise=True
    )

    # 2. Per-class F1 chart (test)
    paths["per_class_f1"] = plot_per_class_f1(test_result, reports_dir)

    # 3. JSON summary
    paths["json_summary"] = save_json_summary(
        test_result, val_result, reports_dir
    )

    # 4. HTML report (embeds all images as base64)
    paths["html_report"] = generate_html_report(
        test_result, val_result,
        cm_path=  paths["confusion_matrix"],
        f1_path=  paths["per_class_f1"],
        save_dir= reports_dir,
    )

    logger.info(
        f"All reports saved to {reports_dir} | "
        f"files: {[p.name for p in paths.values()]}"
    )
    return paths
