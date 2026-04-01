"""
Generate LaTeX table for prefill phrasing sensitivity analysis.

Creates a publication-ready table showing how different prefill phrases affect performance:
- CoT, S2, and variant phrases (o2, artifacts, style, details, flaws)
- Sorted by average Macro F1 ascending
- Deltas relative to CoT
- Best result per dataset column bolded
- Columns: Prefill Phrase, D3, DF40, GenImage
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared configuration
import plot_config as pc
import config

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset configuration (2k variants) - use same order as plot_config
DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
DATASET_DISPLAY_NAMES = {
    'd3-2k': 'D3 (2k)',
    'df40-2k': 'DF40 (2k)',
    'genimage-2k': 'GenImage (2k)'
}

# All phrases to include in the table
ALL_PHRASES = ['cot', 's2', 'o2', 'artifacts', 'style', 'details', 'flaws']

# Models to generate tables for
MODELS = pc.MODEL_ORDER  # ['llava-onevision-7b', 'qwen25-vl-7b', 'qwen3-vl-8b']

# Output directory
TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_phrase_text(phrase_name: str) -> str:
    """Get the full prefill text for a phrase."""
    return config.get_phrase_text(phrase_name)


def find_latest_performance_file(dataset: str, model: str, phrase: str, mode: str = 'prefill') -> Optional[Path]:
    """Find the latest performance JSON file for given configuration."""
    base_dir = Path(__file__).resolve().parent.parent / "output" / dataset / model

    if phrase == 'baseline':
        search_dir = base_dir / 'baseline'
    else:
        search_dir = base_dir / phrase / mode

    if not search_dir.exists():
        return None

    # Find all performance_fixed JSON files (with null handling)
    performance_files = list(search_dir.glob('performance_fixed_*.json'))

    if not performance_files:
        return None

    # Return the most recent file
    return max(performance_files, key=lambda p: p.stat().st_mtime)


def load_macro_f1(dataset: str, model: str, phrase: str, mode: str = 'prefill') -> Optional[float]:
    """Load macro F1 score from performance JSON file."""
    perf_file = find_latest_performance_file(dataset, model, phrase, mode)

    if perf_file is None:
        return None

    try:
        with open(perf_file, 'r') as f:
            data = json.load(f)
        return data['metrics']['macro_f1']
    except (KeyError, json.JSONDecodeError, FileNotFoundError):
        return None


def format_f1_with_delta(f1: Optional[float], cot_f1: Optional[float], is_bold: bool = False) -> str:
    """Format F1 score with delta annotation relative to CoT."""
    if f1 is None:
        return "---"

    # Convert to percentage and round for display
    f1_pct = f1 * 100
    f1_rounded = round(f1_pct, 1)
    f1_str = f"{f1_rounded:.1f}"

    if is_bold:
        f1_str = f"\\textbf{{{f1_str}}}"

    # Calculate delta if CoT baseline exists
    if cot_f1 is not None and cot_f1 != f1:
        cot_pct = cot_f1 * 100
        cot_rounded = round(cot_pct, 1)
        # Calculate delta from rounded values for consistency with displayed numbers
        delta = f1_rounded - cot_rounded
        if abs(delta) >= 0.05:  # Only show delta if >= 0.05
            sign = "+" if delta > 0 else ""
            delta_str = f"{{\\fontsize{{4.75}}{{5.75}}\\selectfont ({sign}{delta:.1f})}}"
            return f"{f1_str} {delta_str}"

    return f1_str


def format_prefill_text(text: str) -> str:
    """Format prefill text using \\figureinput macro."""
    if text == '---':
        return '---'
    # Escape special LaTeX characters first
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return f"{{\\fontsize{{6pt}}{{7.2pt}}\\selectfont\\figureinput{{{text}}}}}"


def generate_latex_table(model: str, output_path: Path):
    """Generate LaTeX table for prefill sensitivity analysis."""

    # Collect scores for all phrases
    rows = []  # List of (phrase_name, prefill_text, {dataset: f1})
    for phrase in ALL_PHRASES:
        prefill_text = get_phrase_text(phrase)
        scores = {}
        for dataset in DATASETS:
            scores[dataset] = load_macro_f1(dataset, model, phrase, 'prefill')
        rows.append((phrase, prefill_text, scores))

    # Separate CoT (pinned first) from the rest
    cot_row = rows[0]  # First entry is 'cot'
    cot_scores = cot_row[2]
    rest = rows[1:]

    # Compute average F1 for sorting (treat None as 0)
    def avg_f1(scores):
        vals = [v for v in scores.values() if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # Sort remaining rows by average F1 ascending
    rest.sort(key=lambda r: avg_f1(r[2]))
    rows = [cot_row] + rest

    # Find best (max) F1 per dataset column
    best_per_dataset = {}
    for dataset in DATASETS:
        valid = [r[2][dataset] for r in rows if r[2][dataset] is not None]
        best_per_dataset[dataset] = max(valid) if valid else None

    # Build LaTeX table
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\fontsize{8.25}{9.75}\\selectfont")
    lines.append("\\setlength{\\tabcolsep}{8pt}")
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")

    # Header: Prefill Phrase, D3, DF40, GenImage
    header = "\\textbf{Prefill Phrase} & "
    header += " & ".join([f"\\textbf{{{DATASET_DISPLAY_NAMES[d]}}}" for d in DATASETS])
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for phrase, prefill_text, scores in rows:
        prefill_formatted = format_prefill_text(prefill_text)
        row = f"{prefill_formatted} & "

        f1_cells = []
        for dataset in DATASETS:
            f1 = scores[dataset]
            cot_f1 = cot_scores[dataset]
            is_best = (f1 is not None and best_per_dataset[dataset] is not None
                       and round(f1 * 100, 1) == round(best_per_dataset[dataset] * 100, 1))
            # CoT row gets no delta (it's the reference)
            ref = None if phrase == 'cot' else cot_f1
            f1_cells.append(format_f1_with_delta(f1, ref, is_bold=is_best))

        row += " & ".join(f1_cells)
        row += " \\\\"
        lines.append(row)

    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Sensitivity to prefill phrasing. "
                "All methods use prefill mode. "
                "Deltas shown relative to CoT. "
                f"Results for {pc.MODEL_NAMES.get(model, model)}.}}")
    lines.append("\\label{tab:prefill_sensitivity}")
    lines.append("\\end{table*}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"LaTeX table generated: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate LaTeX tables for all models."""
    for model in MODELS:
        output_path = TABLES_DIR / f"prefill_sensitivity_{model}.tex"
        generate_latex_table(model, output_path)


if __name__ == '__main__':
    main()
