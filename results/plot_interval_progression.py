"""
Plot Interval Progression: Avg Token Probability & Macro F1

Creates a publication-ready line plot showing how avg_token_prob and macro_f1
evolve across reasoning intervals (0%, 25%, 50%, 75%, 100%) for different
phrase prompting methods.

Features:
- 6 subplots (2 rows × 3 columns): Qwen and LLaVa across 3 datasets
- 10 lines per subplot: baseline, cot (prefill/prompt), s2 (prefill/prompt)
- Solid lines for avg_token_prob, dotted lines for macro_f1
- Circle markers for prefill, square markers for prompt
- Simplified legend with caption note
- Colorblind-friendly palette from plot_config

Output: interval_progression.pdf in figures/
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared config
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Colors (Paul Tol colorblind-friendly from sunset discrete colormap)
BASELINE_COLOR = '#364B9A'  # Blue (pos 0)
PREFILL_COLOR = '#DD3D2D'   # Red (pos 9)
PROMPT_COLOR = '#C2E4EF'    # Light cyan (pos 4)

# Figure dimensions (ACL format - 2 rows per model: 1 for avg_prob, 1 for macro_f1)
FIGURE_HEIGHT = 4.0  # inches

# Output configuration (will create separate files for each model)
OUTPUT_DIR = pc.FIGURES_DIR

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_intervals(df_row):
    """
    Extract 5-element arrays for avg_prob and macro_f1 from a dataframe row.

    Args:
        df_row: Single row from aggregate_intervals.csv

    Returns:
        Tuple of (avg_probs, macro_f1s) as numpy arrays
    """
    intervals = [0.00, 0.25, 0.50, 0.75, 1.00]
    avg_probs = [df_row[f'avg_prob_{i:.2f}'] for i in intervals]
    macro_f1s = [df_row[f'macro_f1_{i:.2f}'] for i in intervals]
    return np.array(avg_probs), np.array(macro_f1s)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_interval_progression(df, model_name, dataset_name, ax, metric='avg_prob'):
    """
    Plot metric progression for one model/dataset combination.
    Averages CoT and S2 for prefill and prompt modes.

    Args:
        df: Filtered dataframe with interval metrics
        model_name: Model identifier (e.g., 'qwen25-vl-7b')
        dataset_name: Dataset identifier (e.g., 'd3-2k')
        ax: Matplotlib axis object
        metric: 'avg_prob' or 'macro_f1'
    """
    x_values = [0, 25, 50, 75, 100]

    # Filter data for this model and dataset
    df_subset = df[(df['model'] == model_name) & (df['dataset'] == dataset_name)]

    # Plot baseline (teal)
    baseline_row = df_subset[(df_subset['phrase'] == 'baseline') & (df_subset['mode'] == 'prefill')]
    if not baseline_row.empty:
        avg_probs, macro_f1s = extract_intervals(baseline_row.iloc[0])
        values = avg_probs if metric == 'avg_prob' else macro_f1s
        ax.plot(x_values, values, color=BASELINE_COLOR, linestyle='-',
               marker='o', markersize=pc.MARKER_SIZE, linewidth=pc.LINE_WIDTH)

    # Plot averaged prefill (average of CoT and S2) - coral
    cot_prefill = df_subset[(df_subset['phrase'] == 'cot') & (df_subset['mode'] == 'prefill')]
    s2_prefill = df_subset[(df_subset['phrase'] == 's2') & (df_subset['mode'] == 'prefill')]

    if not cot_prefill.empty and not s2_prefill.empty:
        cot_avg_probs, cot_macro_f1s = extract_intervals(cot_prefill.iloc[0])
        s2_avg_probs, s2_macro_f1s = extract_intervals(s2_prefill.iloc[0])

        if metric == 'avg_prob':
            values = (cot_avg_probs + s2_avg_probs) / 2
        else:
            values = (cot_macro_f1s + s2_macro_f1s) / 2

        ax.plot(x_values, values, color=PREFILL_COLOR, linestyle='-',
               marker='o', markersize=pc.MARKER_SIZE, linewidth=pc.LINE_WIDTH)

    # Plot averaged prompt (average of CoT and S2) - purple
    cot_prompt = df_subset[(df_subset['phrase'] == 'cot') & (df_subset['mode'] == 'prompt')]
    s2_prompt = df_subset[(df_subset['phrase'] == 's2') & (df_subset['mode'] == 'prompt')]

    if not cot_prompt.empty and not s2_prompt.empty:
        cot_avg_probs, cot_macro_f1s = extract_intervals(cot_prompt.iloc[0])
        s2_avg_probs, s2_macro_f1s = extract_intervals(s2_prompt.iloc[0])

        if metric == 'avg_prob':
            values = (cot_avg_probs + s2_avg_probs) / 2
        else:
            values = (cot_macro_f1s + s2_macro_f1s) / 2

        ax.plot(x_values, values, color=PROMPT_COLOR, linestyle='-',
               marker='o', markersize=pc.MARKER_SIZE, linewidth=pc.LINE_WIDTH)


def compute_subplot_range(df, model_name, dataset_name, metric='avg_prob'):
    """
    Compute min/max range for a specific model/dataset/metric combination.

    Args:
        df: Filtered dataframe with interval metrics
        model_name: Model identifier
        dataset_name: Dataset identifier
        metric: 'avg_prob' or 'macro_f1'

    Returns:
        Tuple of (y_min, y_max) with padding
    """
    # Filter data for this specific model and dataset
    df_subset = df[(df['model'] == model_name) & (df['dataset'] == dataset_name)]

    all_values = []
    intervals = [0.00, 0.25, 0.50, 0.75, 1.00]

    for _, row in df_subset.iterrows():
        for interval in intervals:
            val = row[f'{metric}_{interval:.2f}']
            if pd.notna(val) and not np.isnan(val):
                all_values.append(val)

    if not all_values:
        return (0, 1)

    min_val = np.min(all_values)
    max_val = np.max(all_values)

    # Add 2.5% padding (exact, not rounded)
    padding = (max_val - min_val) * 0.025
    y_min = max(0, min_val - padding)
    y_max = min(1, max_val + padding)

    return (y_min, y_max)


def create_custom_legend():
    """
    Create custom legend handles for simplified 3-entry legend.
    Shows baseline, prefill (avg of CoT+S2), and prompt (avg of CoT+S2).

    Returns:
        Tuple of (handles, labels) for fig.legend()
    """
    legend_labels = ['Baseline', 'Prefill', 'Prompt']

    # Create dummy lines with correct styling - three different colors
    baseline_line = plt.Line2D([0], [0], color=BASELINE_COLOR, marker='o',
                              linestyle='-', linewidth=pc.LINE_WIDTH, markersize=pc.MARKER_SIZE)
    prefill_line = plt.Line2D([0], [0], color=PREFILL_COLOR, marker='o',
                             linestyle='-', linewidth=pc.LINE_WIDTH, markersize=pc.MARKER_SIZE)
    prompt_line = plt.Line2D([0], [0], color=PROMPT_COLOR, marker='o',
                            linestyle='-', linewidth=pc.LINE_WIDTH, markersize=pc.MARKER_SIZE)

    legend_handles = [baseline_line, prefill_line, prompt_line]

    return legend_handles, legend_labels

# =============================================================================
# MAIN
# =============================================================================

def plot_model_figure(df, model_name, model_display_name, datasets, dataset_names_display):
    """
    Create a figure for one model with 2 rows × 3 columns.

    Args:
        df: Filtered dataframe
        model_name: Model identifier (e.g., 'qwen25-vl-7b')
        model_display_name: Display name for model
        datasets: List of dataset names
        dataset_names_display: List of display names for datasets

    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT), sharex=True)

    print(f"\n📊 Creating {model_display_name} figure...")

    # Row 0: Avg Token Probability
    print(f"  Row 0: {model_name} - Avg Token Probability")
    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        plot_interval_progression(df, model_name, dataset, ax, metric='avg_prob')

        # Compute tight y-limits for this specific subplot
        y_min, y_max = compute_subplot_range(df, model_name, dataset, metric='avg_prob')

        # Styling
        ax.set_xlim(0, 100)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
        ax.grid(axis='both', linestyle=pc.GRID_LINESTYLE, alpha=pc.GRID_ALPHA,
               color=pc.GRID_COLOR, linewidth=pc.GRID_LINEWIDTH)

        # Titles (only top row)
        ax.set_title(dataset_names_display[col_idx], fontsize=pc.TITLE_FONT_SIZE, pad=5)

        # Y-labels (only left column)
        if col_idx == 0:
            ax.set_ylabel('Answer Confidence', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

        print(f"    ✓ {dataset}")

    # Row 1: Macro F1
    print(f"  Row 1: {model_name} - Macro F1")
    for col_idx, dataset in enumerate(datasets):
        ax = axes[1, col_idx]
        plot_interval_progression(df, model_name, dataset, ax, metric='macro_f1')

        # Compute tight y-limits for this specific subplot
        y_min, y_max = compute_subplot_range(df, model_name, dataset, metric='macro_f1')

        # Styling
        ax.set_xlim(0, 100)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0\%', '25\%', '50\%', '75\%', '100\%'], fontsize=pc.TICK_LABEL_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
        ax.grid(axis='both', linestyle=pc.GRID_LINESTYLE, alpha=pc.GRID_ALPHA,
               color=pc.GRID_COLOR, linewidth=pc.GRID_LINEWIDTH)

        # Y-labels (only left column)
        if col_idx == 0:
            ax.set_ylabel('Macro F1', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        # X-label (only bottom middle)
        if col_idx == 1:
            ax.set_xlabel('Reasoning Interval', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

        print(f"    ✓ {dataset}")

    # Adjust layout with precise control over spacing
    left_margin = 0.07
    right_margin = 0.98
    wspace = 0.15
    plt.subplots_adjust(top=0.90, bottom=0.10, left=left_margin, right=right_margin,
                       hspace=0.05, wspace=wspace)

    # Calculate the center of the middle column for legend positioning
    middle_column_center = pc.calculate_middle_column_center(left_margin, right_margin, wspace, n_columns=3)

    # Create custom legend centered on middle column
    print(f"  🎨 Adding legend...")
    legend_handles, legend_labels = create_custom_legend()
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3,
              fontsize=pc.LEGEND_FONT_SIZE, frameon=False,
              bbox_to_anchor=(middle_column_center, 1.0))

    return fig


def main():
    """Main plotting pipeline."""
    print("="*80)
    print("INTERVAL PROGRESSION PLOTS (Separate per Model)")
    print("="*80)

    # Enable LaTeX rendering for publication quality
    pc.set_publication_style()

    # Load data
    csv_path = Path(__file__).parent / "aggregate_intervals.csv"
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        print("   Please run aggregate_margins.py first to generate the CSV.")
        return

    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'OK']  # Only successful runs
    print(f"✓ Loaded {len(df)} configurations from {csv_path}")

    datasets = ['d3-2k', 'df40-2k', 'genimage-2k']
    dataset_names_display = ['D3', 'DF40', 'GenImage']

    models = [
        ('qwen25-vl-7b', 'Qwen2.5-VL-7B'),
        ('llava-onevision-7b', 'LLaVa-OneVision-7B'),
        ('llama32-vision-11b', 'Llama-3.2-11B-Vision')
    ]

    # Create separate figure for each model
    for model_name, model_display_name in models:
        fig = plot_model_figure(df, model_name, model_display_name, datasets, dataset_names_display)

        # Save both PDF and PNG
        output_base = OUTPUT_DIR / f"interval_progression_{model_name.split('-')[0]}"
        print(f"  💾 Saving {output_base}...")
        pdf_path, png_path = pc.save_figure(fig, output_base)
        plt.close(fig)

        print(f"  ✅ Saved {model_display_name} figure (PDF + PNG)")

    print("\n" + "="*80)
    print("✅ INTERVAL PROGRESSION PLOTS COMPLETED")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files created (PDF + PNG):")
    print(f"  - interval_progression_qwen25 (Qwen2.5-VL-7B)")
    print(f"  - interval_progression_llava (LLaVa-OneVision-7B)")
    print(f"  - interval_progression_llama32 (Llama-3.2-11B-Vision)")
    print(f"Formats: PDF + PNG (300 DPI)")
    print(f"Size: {pc.ACL_FULL_WIDTH} × {FIGURE_HEIGHT} inches per figure")
    print(f"Subplots: 2 rows × 3 columns (6 total) per figure")
    print(f"  Row 0: Avg Token Probability (D3, DF40, GenImage)")
    print(f"  Row 1: Macro F1 (D3, DF40, GenImage)")
    print(f"Lines per subplot: 3 (baseline, prefill avg, prompt avg)")
    print(f"  Prefill/Prompt = average of CoT and S2")
    print(f"Y-axis: Individual tight ranges per subplot (5% padding)")
    print(f"Tick label size: {pc.TICK_LABEL_FONT_SIZE}pt (both x and y axes)")
    print("="*80)


if __name__ == "__main__":
    main()
