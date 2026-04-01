"""
Response Length Forest Plot

Horizontal forest plot showing token count distributions per mode.
- 3 subplots (D3, DF40, GenImage) arranged horizontally
- Y-axis: 3 models × 4 rows = 12 rows (baseline + pooled prefill/prompt/pseudo)
- X-axis: token count
- Square marker at median, annotated with value
- Error bars: IQR (p25–p75)
- Color = mode

Output: response_length_forest.pdf in figures/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_config as pc

# ============================================================================
# CONFIGURATION
# ============================================================================

FIGURE_HEIGHT = 2.6

# 4 rows per model: (method_key, mode_key, display_label)
ROW_SLOTS = [
    ('Baseline', '-',                     'None'),
    ('Pooled',   'prompt',               'Prompt'),
    ('Pooled',   'prefill-pseudo-system', 'Pseudo'),
    ('Pooled',   'prefill',              'Prefill'),
]

ROWS_PER_MODEL = len(ROW_SLOTS)

# Colors per mode (from plot_config, with '-' alias for baseline)
MODE_COLORS = {'-': pc.MODE_COLORS['none'], **{k: v for k, v in pc.MODE_COLORS.items() if k != 'none'}}

# Marker
MARKER_SIZE = 8

# Dataset config (use 2k versions, display without suffix)
DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
DATASET_NAMES = {'d3-2k': 'D3', 'df40-2k': 'DF40', 'genimage-2k': 'GenImage'}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_response_length_data():
    """Load response_length.json into a lookup dict keyed by (dataset, model, method, mode)."""
    data_path = Path(__file__).resolve().parent / "response_length.json"
    with open(data_path) as f:
        rows = json.load(f)

    lookup = {}
    for row in rows:
        key = (row['dataset'], row['model'], row['method'], row['mode'])
        lookup[key] = row
    return lookup


# ============================================================================
# PLOTTING
# ============================================================================

def create_forest_plot(data):
    pc.set_publication_style()

    n_models = len(pc.MODEL_ORDER)
    n_rows = n_models * ROWS_PER_MODEL
    model_gap = 1.0  # extra gap between model groups

    fig, axes = plt.subplots(1, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT), sharey=True)

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]

        y_positions = []
        y_labels = []
        current_y = 0

        for model_idx, model in enumerate(pc.MODEL_ORDER):
            if model_idx > 0:
                current_y += model_gap

            group_start = current_y

            # Look up Prompt median for this model/dataset
            prompt_key = (dataset, model, 'Pooled', 'prompt')
            prompt_row = data.get(prompt_key)
            prompt_median = prompt_row['median_tokens'] if prompt_row else None

            for slot_idx, (method, mode, label) in enumerate(ROW_SLOTS):
                y_pos = current_y
                y_positions.append(y_pos)
                y_labels.append(label)

                key = (dataset, model, method, mode)
                row = data.get(key)

                if row is None:
                    current_y += 1
                    continue

                median = row['median_tokens']
                p25 = row['p25']
                p75 = row['p75']

                # Color by mode
                color = MODE_COLORS.get(mode, '#888888')

                # Error bar (IQR)
                ax.errorbar(
                    median, y_pos,
                    xerr=[[median - p25], [p75 - median]],
                    fmt='none',
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=2,
                    capthick=0.8,
                )

                # Square marker at median
                ax.plot(
                    median, y_pos,
                    marker='s',
                    markersize=MARKER_SIZE,
                    color=color,
                    markeredgecolor='black',
                    markeredgewidth=0.4,
                    zorder=5,
                )

                # Annotate with median value inside marker
                ax.annotate(
                    f'{median:.0f}',
                    xy=(median, y_pos),
                    fontsize=pc.ANNOTATION_FONT_SIZE,
                    va='center',
                    ha='center',
                    color='white',
                    fontweight='bold',
                    zorder=6,
                )

                # Annotate relative % difference from Prompt (for Pseudo and Prefill)
                if mode in ('prefill-pseudo-system', 'prefill') and prompt_median:
                    pct_diff = (median - prompt_median) / prompt_median * 100
                    ax.annotate(
                        f'{pct_diff:+.0f}\\%',
                        xy=(p75, y_pos),
                        xytext=(4, 0),
                        textcoords='offset points',
                        fontsize=pc.ANNOTATION_FONT_SIZE,
                        va='center',
                        ha='left',
                        color='black',
                    )

                current_y += 1

            # Model label only on first subplot
            if ax_idx == 0:
                group_center = group_start + (ROWS_PER_MODEL - 1) / 2
                ax.text(
                    -0.05, group_center,
                    pc.MODEL_NAMES[model],
                    transform=ax.get_yaxis_transform(),
                    fontsize=pc.AXIS_LABEL_FONT_SIZE,
                    fontweight='bold',
                    va='center',
                    ha='center',
                    rotation=90,
                )

        # Horizontal separator lines between model groups
        sep_y = 0
        for model_idx in range(n_models):
            if model_idx > 0:
                sep_y_pos = sep_y - model_gap / 2
                ax.axhline(y=sep_y_pos, color='gray', linewidth=0.5, linestyle='-', alpha=0.4)
            sep_y += ROWS_PER_MODEL + (model_gap if model_idx < n_models - 1 else 0)

        # Axes config — no y-tick labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

        ax.set_xlabel('')
        ax.text(0.5, 1.02, DATASET_NAMES[dataset],
                ha='center', va='bottom', fontsize=pc.TITLE_FONT_SIZE,
                transform=ax.transAxes)
        ax.tick_params(axis='x', labelsize=pc.TICK_LABEL_FONT_SIZE)
        ax.grid(axis='x', linestyle=pc.GRID_LINESTYLE, alpha=pc.GRID_ALPHA,
                color=pc.GRID_COLOR, linewidth=pc.GRID_LINEWIDTH)
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

    # Legend - one entry per row type
    legend_items = [
        ('-',                     'None'),
        ('prompt',               'Prompt'),
        ('prefill-pseudo-system', 'Pseudo'),
        ('prefill',              'Prefill'),
    ]
    all_handles = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=MODE_COLORS[mode_key],
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=7, label=label)
        for mode_key, label in legend_items
    ]

    plt.subplots_adjust(
        top=0.87, bottom=0.13, left=0.03, right=0.99, wspace=0.0
    )

    fig.text(0.515, 0.01, 'Reasoning Tokens', ha='center', va='bottom',
             fontsize=pc.AXIS_LABEL_FONT_SIZE)

    fig.legend(
        handles=all_handles,
        loc='upper center',
        ncol=len(all_handles),
        fontsize=pc.LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    # Save
    output_base = pc.FIGURES_DIR / "response_length_forest"
    pdf_path, png_path = pc.save_figure(fig, output_base)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


def main():
    print("Loading response length data...")
    data = load_response_length_data()
    print(f"Loaded {len(data)} entries")

    print("Creating forest plot...")
    create_forest_plot(data)
    print("Done!")


if __name__ == "__main__":
    main()
