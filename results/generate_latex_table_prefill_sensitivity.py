"""
Generate LaTeX table for prefill phrasing sensitivity analysis.

Creates a publication-ready table showing how different prefill phrases affect performance:
- Baseline (no phrase)
- CoT (chain of thought)
- S2 (style and synthesis)
- Variants (o2, artifacts, style, details, flaws) with deltas relative to S2
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

# Phrase configuration
PHRASES = {
    'baseline': {'display': 'Baseline', 'prefill': '---'},
    'cot': {'display': 'CoT', 'prefill': None},  # Will load from config
    's2': {'display': 'S2', 'prefill': None},    # Will load from config
    'variants': [
        'o2',
        'artifacts',
        'style',
        'details',
        'flaws'
    ]
}

# Model to use for the table
MODEL = 'qwen25-vl-7b'

# Output directory
TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_phrase_text(phrase_name: str) -> str:
    """Get the full prefill text for a phrase."""
    if phrase_name == 'baseline':
        return '---'
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

    # Find all performance JSON files
    performance_files = list(search_dir.glob('performance_*.json'))

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


def format_f1_with_delta(f1: Optional[float], baseline_f1: Optional[float]) -> str:
    """Format F1 score with delta annotation relative to baseline."""
    if f1 is None:
        return "---"

    # Convert to percentage
    f1_pct = f1 * 100
    f1_str = f"{f1_pct:.1f}"

    # Calculate delta if baseline exists
    if baseline_f1 is not None and baseline_f1 != f1:
        baseline_pct = baseline_f1 * 100
        delta = f1_pct - baseline_pct
        if abs(delta) >= 0.05:  # Only show delta if >= 0.05
            sign = "+" if delta > 0 else ""
            delta_str = f"{{\\scriptsize ({sign}{delta:.1f})}}"
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
    return f"\\figureinput{{{text}}}"


def generate_latex_table(model: str, output_path: Path):
    """Generate LaTeX table for prefill sensitivity analysis."""

    # Load phrase texts
    PHRASES['cot']['prefill'] = get_phrase_text('cot')
    PHRASES['s2']['prefill'] = get_phrase_text('s2')

    # Collect all data
    data = {}

    # Baseline
    data['baseline'] = {}
    for dataset in DATASETS:
        f1 = load_macro_f1(dataset, model, 'baseline')
        data['baseline'][dataset] = f1

    # CoT
    data['cot'] = {}
    for dataset in DATASETS:
        f1 = load_macro_f1(dataset, model, 'cot', 'prefill')
        data['cot'][dataset] = f1

    # S2
    data['s2'] = {}
    for dataset in DATASETS:
        f1 = load_macro_f1(dataset, model, 's2', 'prefill')
        data['s2'][dataset] = f1

    # Variants
    data['variants'] = {}
    for variant in PHRASES['variants']:
        data['variants'][variant] = {
            'prefill': get_phrase_text(variant),
            'scores': {}
        }
        for dataset in DATASETS:
            f1 = load_macro_f1(dataset, model, variant, 'prefill')
            data['variants'][variant]['scores'][dataset] = f1

    # Start building LaTeX table
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{llllll}")
    lines.append("\\toprule")

    # Header
    header = "\\textbf{Phrase} & \\textbf{Prefill} & "
    header += " & ".join([f"\\textbf{{{DATASET_DISPLAY_NAMES[d]}}}" for d in DATASETS])
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Baseline row
    row = "Baseline & --- & "
    row += " & ".join([format_f1_with_delta(data['baseline'][d], None) for d in DATASETS])
    row += " \\\\"
    lines.append(row)
    lines.append("\\midrule")

    # CoT row
    prefill_text = format_prefill_text(PHRASES['cot']['prefill'])
    row = f"CoT & {prefill_text} & "
    row += " & ".join([format_f1_with_delta(data['cot'][d], None) for d in DATASETS])
    row += " \\\\"
    lines.append(row)
    lines.append("\\midrule")

    # S2 row
    prefill_text = format_prefill_text(PHRASES['s2']['prefill'])
    row = f"S2 & {prefill_text} & "
    row += " & ".join([format_f1_with_delta(data['s2'][d], None) for d in DATASETS])
    row += " \\\\"
    lines.append(row)
    lines.append("\\midrule")

    # Variants rows (nested with multirow)
    num_variants = len(PHRASES['variants'])
    for i, variant in enumerate(PHRASES['variants']):
        variant_data = data['variants'][variant]
        prefill_text = format_prefill_text(variant_data['prefill'])

        row = ""
        if i == 0:
            # First variant row: include multirow label
            row += f"\\multirow{{{num_variants}}}{{*}}{{Variants}} & "
        else:
            # Subsequent rows: empty phrase column
            row += " & "

        # Prefill column
        row += f"{prefill_text} & "

        # Dataset columns with deltas relative to S2
        f1_values = []
        for dataset in DATASETS:
            f1 = variant_data['scores'][dataset]
            s2_f1 = data['s2'][dataset]
            f1_values.append(format_f1_with_delta(f1, s2_f1))

        row += " & ".join(f1_values)
        row += " \\\\"
        lines.append(row)

    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Sensitivity to prefill phrasing. "
                "All methods use prefill mode. "
                "Variants show performance change relative to S2 baseline. "
                f"Results for {MODEL}.}}")
    lines.append("\\label{tab:prefill_sensitivity}")
    lines.append("\\end{table*}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ LaTeX table generated: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate LaTeX table."""
    output_path = TABLES_DIR / f"prefill_sensitivity_{MODEL}.tex"
    generate_latex_table(MODEL, output_path)


if __name__ == '__main__':
    main()
