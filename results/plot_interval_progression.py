"""
Plot Interval Progression: Answer Confidence

Creates publication-ready figures showing how answer confidence (avg_token_prob)
evolve across reasoning intervals (0%–100%) for 7 lines:
None (baseline), plus CoT (dotted) and S2 (solid) for Prompt, Pseudo, Prefill.

Features:
- 9 subplots per figure (3 rows x 3 columns: 3 models x 3 datasets)
- 7 lines per subplot: None + 3 modes x 2 phrases (CoT dotted, S2 solid)
- Colors from plot_config MODE_COLORS
- Loads directly from performance_*_with_conf.json files

Output: interval_progression.pdf, interval_progression_f1.pdf in figures/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

FIGURE_HEIGHT = 5.0  # inches

# 4 averaged lines: baseline + 3 modes (each averaging CoT and S2)
# Colors from plot_config MODE_COLORS
LINE_DEFS = [
    ('None',    pc.MODE_COLORS['none'],                  [('baseline', 'prefill')]),
    ('Prompt',  pc.MODE_COLORS['prompt'],                [('cot', 'prompt'), ('s2', 'prompt')]),
    ('Pseudo',  pc.MODE_COLORS['prefill-pseudo-system'], [('cot', 'prefill-pseudo-system'), ('s2', 'prefill-pseudo-system')]),
    ('Prefill', pc.MODE_COLORS['prefill'],               [('cot', 'prefill'), ('s2', 'prefill')]),
]

# All (phrase, mode) combos we need to load
ALL_SLOTS = set()
for _, _, slots in LINE_DEFS:
    ALL_SLOTS.update(slots)

MODELS = [
    ('llava-onevision-7b', 'LLaVA'),
    ('qwen25-vl-7b', 'Qwen2.5'),
    ('qwen3-vl-8b', 'Qwen3'),
]

DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
DATASET_DISPLAY = ['D3 (2k)', 'DF40 (2k)', 'GenImage (2k)']

INTERVALS = ['0.00', '0.25', '0.50', '0.75', '1.00']
X_VALUES = [0, 25, 50, 75, 100]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_interval_data(dataset, model, phrase, mode):
    """
    Load interval metrics from performance_*_with_conf.json.

    Returns:
        Dict with 'avg_token_prob' and 'macro_f1' keys, each a 5-element
        numpy array (raw fractions). Returns None if not found.
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n=1)

    perf_files = sorted(output_dir.glob("performance_*_with_conf.json"))
    if not perf_files:
        return None

    with open(perf_files[-1], 'r') as f:
        data = json.load(f)

    intervals = data.get('intervals', {})
    if not intervals:
        return None

    avg_probs = []
    for iv in INTERVALS:
        iv_data = intervals.get(iv, {})
        avg_probs.append(iv_data.get('avg_token_prob_stats', {}).get('mean', 0.0))

    return np.array(avg_probs)


def collect_model_data(model):
    """
    Load raw interval data, then average CoT+S2 per mode.

    Returns:
        Nested dict: {dataset: [(label, color, data_dict), ...]}
        where data_dict has 'avg_token_prob' and 'macro_f1' arrays.
    """
    result = {}
    for dataset in DATASETS:
        raw = {}
        for phrase, mode in ALL_SLOTS:
            try:
                data = load_interval_data(dataset, model, phrase, mode)
                if data is not None:
                    raw[(phrase, mode)] = data
            except Exception as e:
                print(f"  Warning: {dataset}/{model}/{phrase}/{mode}: {e}")

        lines = []
        for label, color, slots in LINE_DEFS:
            available = [raw[s] for s in slots if s in raw]
            if not available:
                continue
            avg_prob = np.mean(available, axis=0)
            lines.append((label, color, avg_prob))

        result[dataset] = lines
    return result


# =============================================================================
# PLOTTING
# =============================================================================

def plot_lines(ax, lines):
    """Plot averaged lines on a single axis."""
    for label, color, avg_prob in lines:
        values = avg_prob * 100
        ax.plot(X_VALUES, values,
                color=color, linestyle='-',
                marker='o', markersize=pc.MARKER_SIZE, linewidth=pc.LINE_WIDTH)


def compute_ylim(lines):
    """Compute tight y-axis limits."""
    all_values = []
    for _, _, avg_prob in lines:
        all_values.extend(avg_prob * 100)

    if not all_values:
        return (0, 100)

    min_val = np.min(all_values)
    max_val = np.max(all_values)
    padding = max((max_val - min_val) * 0.05, 1.0)
    return (max(0, round(min_val - padding)), min(100, round(max_val + padding)))


def create_legend():
    """Create legend handles for the 4 lines."""
    return [
        Line2D([0], [0], color=color, linestyle='-', marker='o',
               linewidth=pc.LINE_WIDTH, markersize=pc.MARKER_SIZE, label=label)
        for label, color, _ in LINE_DEFS
    ]


def create_figure(all_model_data):
    """Create 3x3 figure: rows=models, cols=datasets."""
    fig, axes = plt.subplots(3, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT), sharex=True)

    for row_idx, (model_name, model_display) in enumerate(MODELS):
        model_data = all_model_data[model_name]

        for col_idx, (dataset, ds_display) in enumerate(zip(DATASETS, DATASET_DISPLAY)):
            ax = axes[row_idx, col_idx]
            lines = model_data.get(dataset, [])

            plot_lines(ax, lines)

            y_min, y_max = compute_ylim(lines)
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

            ax.set_xlim(0, 100)
            if row_idx == 2:
                ax.set_xticks(X_VALUES)
                ax.set_xticklabels(['0\\%', '25\\%', '50\\%', '75\\%', '100\\%'],
                                   fontsize=pc.TICK_LABEL_FONT_SIZE)

            ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
            ax.grid(axis='both', linestyle=pc.GRID_LINESTYLE, alpha=pc.GRID_ALPHA,
                    color=pc.GRID_COLOR, linewidth=pc.GRID_LINEWIDTH)
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)

            if row_idx == 0:
                ax.set_title(ds_display, fontsize=pc.TITLE_FONT_SIZE, pad=5)
            if col_idx == 0:
                ax.set_ylabel(model_display,
                              fontsize=pc.AXIS_LABEL_FONT_SIZE)
            if row_idx == 2 and col_idx == 1:
                ax.set_xlabel('Partial-response Interval', fontsize=pc.AXIS_LABEL_FONT_SIZE)

    # Layout
    left_margin = 0.10
    right_margin = 0.98
    wspace = 0.15
    plt.subplots_adjust(top=0.92, bottom=0.07, left=left_margin, right=right_margin,
                        hspace=0.15, wspace=wspace)

    # Shared y-axis label
    fig.text(0.01, 0.5, 'Answer Confidence (\\%)',
             va='center', ha='left', rotation='vertical',
             fontsize=pc.AXIS_LABEL_FONT_SIZE)

    handles = create_legend()
    middle_x = pc.calculate_middle_column_center(left_margin, right_margin, wspace, n_columns=3)
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               fontsize=pc.LEGEND_FONT_SIZE, frameon=False,
               bbox_to_anchor=(middle_x, 1.0))

    return fig


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(all_model_data):
    """Print a table of answer confidence values across intervals."""
    intervals_pct = ['0%', '25%', '50%', '75%', '100%']
    header = f"{'Model':<10} {'Dataset':<10} {'Mode':<10} " + \
             " ".join(f"{iv:>6}" for iv in intervals_pct) + f" {'Delta':>7}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("Answer Confidence (%) across Partial-response Intervals")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for model_name, model_display in MODELS:
        model_data = all_model_data[model_name]
        for dataset, ds_display in zip(DATASETS, DATASET_DISPLAY):
            lines = model_data.get(dataset, [])
            for label, color, avg_prob in lines:
                values = avg_prob * 100
                delta = values[-1] - values[0]
                vals_str = " ".join(f"{v:6.1f}" for v in values)
                print(f"{model_display:<10} {ds_display:<10} {label:<10} {vals_str} {delta:>+7.1f}")
            print(sep)


# =============================================================================
# MAIN
# =============================================================================

def main():
    pc.set_publication_style()

    print("Generating Interval Progression Plots...")
    print("Lines: None, Prompt/Pseudo/Prefill x CoT(dotted)/S2(solid)")

    all_model_data = {}
    for model_name, model_display in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_display} ({model_name})")
        print(f"{'='*60}")

        model_data = collect_model_data(model_name)
        all_model_data[model_name] = model_data

        for ds in DATASETS:
            n_lines = len(model_data.get(ds, []))
            labels = [l[0] for l in model_data.get(ds, [])]
            print(f"  {ds}: {n_lines}/4 lines — {', '.join(labels)}")

    fig = create_figure(all_model_data)
    output_base = pc.FIGURES_DIR / "interval_progression"
    pdf_path, png_path = pc.save_figure(fig, output_base)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Print summary table
    print_summary_table(all_model_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
