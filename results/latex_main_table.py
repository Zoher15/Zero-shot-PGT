"""
LaTeX Main Results Table Generator

Creates a single publication-ready LaTeX table with Macro F1 scores:
- Columns: 3 datasets x 3 models (9 data columns)
- Rows: Baseline, CoT (Prompt / Pseudo-System / Prefill), S2 (same)
- Deltas relative to Prompt shown in parentheses
- Best score per dataset-model column bolded

Uses COLM paper format (5.5" textwidth).

Output: results/tables/main_table.tex
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = pc.DATASET_ORDER  # ['d3', 'df40', 'genimage']
DATASET_DISPLAY = pc.DATASET_NAMES

MODELS = pc.MODEL_ORDER  # ['llava-onevision-7b', 'qwen25-vl-7b', 'qwen3-vl-8b']
MODEL_DISPLAY = pc.MODEL_NAMES

N_RESPONSES = pc.DEFAULT_N_RESPONSES

# Row definitions: (phrase, mode, display_name)
# Order: prompt, pseudo-system, prefill
ROWS = [
    ('baseline', 'prefill', 'Baseline'),
    ('cot',      'prompt',                'Prompt'),
    ('cot',      'prefill-pseudo-system', 'Pseudo'),
    ('cot',      'prefill',               'Prefill'),
    ('s2',       'prompt',                'Prompt'),
    ('s2',       'prefill-pseudo-system', 'Pseudo'),
    ('s2',       'prefill',               'Prefill'),
]

TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_macro_f1(dataset: str, model: str, phrase: str, mode: str) -> Optional[float]:
    """Load macro F1 from the latest performance_fixed JSON."""
    output_dir = config.get_output_dir(dataset, model, phrase, mode, N_RESPONSES)
    files = sorted(output_dir.glob("performance_fixed_*.json"))
    if not files:
        return None
    try:
        with open(files[-1]) as f:
            data = json.load(f)
        return data['metrics']['macro_f1']
    except (KeyError, json.JSONDecodeError):
        return None


def collect_all_data() -> Dict:
    """Collect all F1 scores. Returns {(phrase, mode): {(dataset, model): f1}}."""
    data = {}
    for phrase, mode, _ in ROWS:
        key = (phrase, mode)
        if key not in data:
            data[key] = {}
            for dataset in DATASETS:
                for model in MODELS:
                    data[key][(dataset, model)] = load_macro_f1(dataset, model, phrase, mode)
    return data

# =============================================================================
# LATEX GENERATION
# =============================================================================

def format_cell(f1: Optional[float], ref_f1: Optional[float], is_best: bool) -> str:
    """Format a single table cell with optional delta and bolding."""
    if f1 is None:
        return "---"

    val = f1 * 100
    val_str = f"{val:.1f}"

    if is_best:
        val_str = f"\\textbf{{{val_str}}}"

    # Delta relative to reference (prompt)
    if ref_f1 is not None and ref_f1 != f1:
        delta = val - ref_f1 * 100
        if abs(delta) >= 0.05:
            sign = "+" if delta > 0 else ""
            val_str += f" {{\\fontsize{{4.75}}{{5.75}}\\selectfont ({sign}{delta:.1f})}}"

    return val_str


def generate_table():
    """Generate and save a single combined LaTeX table."""
    data = collect_all_data()

    # Column keys: (dataset, model) pairs
    col_keys = [(d, m) for d in DATASETS for m in MODELS]
    n_data_cols = len(col_keys)  # 9

    # Find best F1 per column (across all rows)
    best = {}
    for col in col_keys:
        values = [(key, data[key][col]) for key in data if data[key][col] is not None]
        if values:
            best[col] = max(values, key=lambda x: x[1])[0]

    # Prompt F1 for delta calculations (per phrase)
    prompt_ref = {}
    for phrase in ['cot', 's2']:
        prompt_key = (phrase, 'prompt')
        prompt_ref[phrase] = {col: data[prompt_key][col] for col in col_keys}

    # Build LaTeX
    # 2 label cols + 9 data cols = 11 columns
    col_spec = "ll" + "l" * n_data_cols
    n_models = len(MODELS)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\fontsize{8.25}{9.75}\\selectfont")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Two-level header: datasets on top, models below
    # Dataset header with multicolumn
    header1 = " & "  # phrase col + mode col
    for i, d in enumerate(DATASETS):
        header1 += f" & \\multicolumn{{{n_models}}}{{c}}{{\\textbf{{{DATASET_DISPLAY[d]}}}}}"
    header1 += " \\\\"
    lines.append(header1)

    # Cmidrules under each dataset group
    cmidrules = ""
    for i, d in enumerate(DATASETS):
        start = 3 + i * n_models
        end = start + n_models - 1
        cmidrules += f"\\cmidrule(lr){{{start}-{end}}} "
    lines.append(cmidrules.strip())

    # Model header
    header2 = "\\textbf{Phrase} & \\textbf{Mode}"
    for d in DATASETS:
        for m in MODELS:
            header2 += f" & {MODEL_DISPLAY[m]}"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # --- Baseline row ---
    key = ('baseline', 'prefill')
    cells = []
    for col in col_keys:
        is_best = best.get(col) == key
        cells.append(format_cell(data[key][col], None, is_best))
    lines.append(f"None & --- & {' & '.join(cells)} \\\\")
    lines.append("\\midrule")

    # --- CoT rows (indices 1-3) ---
    cot_rows = ROWS[1:4]
    for i, (phrase, mode, display) in enumerate(cot_rows):
        key = (phrase, mode)
        cells = []
        for col in col_keys:
            is_best = best.get(col) == key
            # No delta for prompt (it's the reference), delta for others
            ref = None if mode == 'prompt' else prompt_ref[phrase][col]
            cells.append(format_cell(data[key][col], ref, is_best))

        phrase_col = f"\\multirow{{{len(cot_rows)}}}{{*}}{{CoT}}" if i == 0 else ""
        lines.append(f"{phrase_col} & {display} & {' & '.join(cells)} \\\\")

    lines.append(f"\\cmidrule(lr){{1-{2 + n_data_cols}}}")

    # --- S2 rows (indices 4-6) ---
    s2_rows = ROWS[4:7]
    for i, (phrase, mode, display) in enumerate(s2_rows):
        key = (phrase, mode)
        cells = []
        for col in col_keys:
            is_best = best.get(col) == key
            ref = None if mode == 'prompt' else prompt_ref[phrase][col]
            cells.append(format_cell(data[key][col], ref, is_best))

        phrase_col = f"\\multirow{{{len(s2_rows)}}}{{*}}{{S2}}" if i == 0 else ""
        lines.append(f"{phrase_col} & {display} & {' & '.join(cells)} \\\\")

    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Macro F1 (\\%) across datasets and models. "
                 "Deltas are relative to Prompt mode within each phrase group. "
                 "Best result per column in \\textbf{bold}.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\end{table*}")

    # Write
    output_path = TABLES_DIR / "main_table.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Saved: {output_path}\n")
    for line in lines:
        print(line)


def print_rankings():
    """Print ranking analysis across all configurations."""
    data = collect_all_data()
    col_keys = [(d, m) for d in DATASETS for m in MODELS]

    # Labels for each configuration
    config_labels = [
        (('baseline', 'prefill'), 'None'),
        (('cot', 'prompt'), 'CoT Prompt'),
        (('cot', 'prefill-pseudo-system'), 'CoT Pseudo'),
        (('cot', 'prefill'), 'CoT Prefill'),
        (('s2', 'prompt'), 'S2 Prompt'),
        (('s2', 'prefill-pseudo-system'), 'S2 Pseudo'),
        (('s2', 'prefill'), 'S2 Prefill'),
    ]

    # Rank each column (1 = best), keyed by (config, col) for safe lookup
    ranks = {key: {} for key, _ in config_labels}
    for col in col_keys:
        scores = [(key, data[key].get(col)) for key, _ in config_labels]
        scores_valid = [(k, s) for k, s in scores if s is not None]
        scores_valid.sort(key=lambda x: x[1], reverse=True)
        for rank, (key, _) in enumerate(scores_valid, 1):
            ranks[key][col] = rank

    # Print overall rankings
    print("\n" + "=" * 70)
    print("OVERALL RANKING (avg rank across 9 dataset-model columns, 1=best)")
    print("=" * 70)
    avg_ranks = []
    for key, label in config_labels:
        r = ranks[key]
        vals = list(r.values())
        avg = sum(vals) / len(vals) if vals else float('inf')
        avg_ranks.append((label, avg, vals))
    avg_ranks.sort(key=lambda x: x[1])
    for label, avg, r in avg_ranks:
        print(f"  {label:<16s}  avg rank: {avg:.2f}  ranks: {r}")

    # Win counts: how many times each config is #1
    print("\n" + "-" * 70)
    print("WIN COUNTS (# of columns where config ranks #1)")
    print("-" * 70)
    for key, label in config_labels:
        wins = sum(1 for r in ranks[key].values() if r == 1)
        if wins > 0:
            print(f"  {label:<16s}  wins: {wins}/9")

    # Mode comparison within each phrase group
    for phrase, phrase_label in [('cot', 'CoT'), ('s2', 'S2')]:
        mode_keys = [
            ((phrase, 'prompt'), 'Prompt'),
            ((phrase, 'prefill-pseudo-system'), 'Pseudo'),
            ((phrase, 'prefill'), 'Prefill'),
        ]
        # Rank only within the 3 modes
        mode_ranks = {key: [] for key, _ in mode_keys}
        for col in col_keys:
            scores = [(key, data[key].get(col)) for key, _ in mode_keys]
            scores_valid = [(k, s) for k, s in scores if s is not None]
            scores_valid.sort(key=lambda x: x[1], reverse=True)
            for rank, (key, _) in enumerate(scores_valid, 1):
                mode_ranks[key].append(rank)

        print(f"\n" + "-" * 70)
        print(f"MODE RANKING WITHIN {phrase_label} (1=best of 3)")
        print("-" * 70)
        for key, label in mode_keys:
            avg = sum(mode_ranks[key]) / len(mode_ranks[key]) if mode_ranks[key] else float('inf')
            wins = sum(1 for r in mode_ranks[key] if r == 1)
            print(f"  {label:<16s}  avg rank: {avg:.2f}  wins: {wins}/9  ranks: {mode_ranks[key]}")

    # Per-model breakdown
    print(f"\n" + "=" * 70)
    print("PER-MODEL AVERAGE RANK (across 3 datasets)")
    print("=" * 70)
    for model in MODELS:
        model_cols = [(d, model) for d in DATASETS]
        print(f"\n  {MODEL_DISPLAY[model]}:")
        model_avgs = []
        for key, label in config_labels:
            model_ranks = [ranks[key][c] for c in model_cols if c in ranks[key]]
            avg = sum(model_ranks) / len(model_ranks) if model_ranks else float('inf')
            model_avgs.append((label, avg))
        model_avgs.sort(key=lambda x: x[1])
        for label, avg in model_avgs:
            print(f"    {label:<16s}  avg rank: {avg:.2f}")


def main():
    print("Generating main results table (COLM format)...\n")
    generate_table()
    print_rankings()
    print("\nDone!")


if __name__ == '__main__':
    main()
