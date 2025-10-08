"""
Self-Consistency Scaling Analysis

Creates publication-ready line plots showing how macro F1 performance scales
with the number of sampled responses (n) across different methods.

Layout:
- 3 subplots (one per dataset: D3, DF40, GenImage)
- X-axis: Number of samples (n = 1, 5, 10, 20)
- Y-axis: Macro F1 score (%)
- Lines: baseline, cot, s2 methods
- Shared legend across subplots

Usage:
    python results/plot_scaling_n.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared config and project config
import config
import plot_config as pc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models to analyze (None = all models, or specify single model)
MODEL_TO_PLOT = None  # None = all models, or specify: 'llava-onevision-7b', 'qwen25-vl-7b', 'llama32-vision-11b'

# Datasets to analyze (2k versions, using order from plot_config)
DATASETS_TO_PLOT = [f'{ds}-2k' for ds in pc.DATASET_ORDER]  # ['d3-2k', 'df40-2k', 'genimage-2k']

# Methods to compare (using order from plot_config)
METHODS_TO_PLOT = pc.METHOD_ORDER  # ['baseline', 'cot', 's2']

# N values to plot
N_VALUES = [1, 5, 10, 20]

# Phrase mode to use
PHRASE_MODE = 'prefill'

# Figure dimensions (ACL format)
FIGURE_HEIGHT = 2.5  # inches

# Output configuration (path will be set per model in create_scaling_plot)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_latest_performance_json(dataset: str, model: str, phrase: str, mode: str, n: int) -> Optional[Dict]:
    """
    Load the most recent performance JSON for given configuration.

    Args:
        dataset: Dataset name (e.g., 'd3-2k')
        model: Model name
        phrase: Phrase name
        mode: Phrase mode
        n: Number of responses

    Returns:
        Performance data dictionary or None if not found
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)

    # Find most recent performance_fixed JSON (with null handling)
    performance_files = sorted(output_dir.glob("performance_fixed_*.json"))
    if not performance_files:
        print(f"⚠️  No performance JSON found for {dataset}/{model}/{phrase}/n={n}")
        return None

    latest_file = performance_files[-1]  # Most recent timestamp

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return data


def collect_scaling_data(model: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Collect macro F1 scores for all n values across datasets and methods.

    Args:
        model: Model name

    Returns:
        Nested dict: {dataset: {method: {n_value: f1_score}}}
    """
    results = {}

    for dataset in DATASETS_TO_PLOT:
        results[dataset] = {}

        for method in METHODS_TO_PLOT:
            results[dataset][method] = {}

            for n in N_VALUES:
                perf_data = load_latest_performance_json(dataset, model, method, PHRASE_MODE, n)

                if perf_data:
                    macro_f1 = perf_data['metrics']['macro_f1']
                    results[dataset][method][n] = macro_f1 * 100  # Convert to percentage
                else:
                    results[dataset][method][n] = None

    return results


def create_scaling_plot(data: Dict, model_name: str):
    """
    Create 3-subplot line plot showing scaling with n.

    Args:
        data: Nested dict {dataset: {method: {n: f1_score}}}
        model_name: Model name for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT), sharey=False)

    # Dataset display names (use plot_config names, strip -2k suffix)
    def get_display_name(dataset: str) -> str:
        base_name = dataset.replace('-2k', '')
        return pc.DATASET_NAMES.get(base_name, dataset)

    for subplot_idx, dataset in enumerate(DATASETS_TO_PLOT):
        ax = axes[subplot_idx]

        # Set subplot title
        ax.set_title(get_display_name(dataset), fontsize=pc.TITLE_FONT_SIZE, pad=5)

        # Prepare data for this dataset
        dataset_results = data[dataset]

        # Calculate y-axis limits FIRST by collecting all values for this dataset
        all_values = []
        for method in METHODS_TO_PLOT:
            method_data = dataset_results[method]
            for n in N_VALUES:
                if method_data[n] is not None:
                    all_values.append(method_data[n])

        # Set y-axis limits based on data range
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            # Round to nearest 5 (floor for min, ceil for max)
            y_min_rounded = np.floor(y_min / 5) * 5
            y_max_rounded = np.ceil(y_max / 5) * 5
            ax.set_ylim(y_min_rounded, y_max_rounded)

        # Plot lines for each method
        for method in METHODS_TO_PLOT:
            method_data = dataset_results[method]

            # Extract n values and scores (filter out None)
            n_vals = []
            f1_scores = []
            for n in N_VALUES:
                if method_data[n] is not None:
                    n_vals.append(n)
                    f1_scores.append(method_data[n])

            if n_vals:  # Only plot if we have data
                ax.plot(
                    n_vals,
                    f1_scores,
                    color=pc.METHOD_COLORS[method],
                    marker='o',
                    linestyle='-',
                    linewidth=pc.LINE_WIDTH,
                    markersize=pc.MARKER_SIZE,
                    label=pc.METHOD_NAMES[method]
                )

        # Configure x-axis
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=pc.TICK_LABEL_FONT_SIZE)
        ax.set_xlim(0, max(N_VALUES) + 1)

        # Set y-axis ticks to multiples of 5
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
        ax.grid(axis='both', linestyle=pc.GRID_LINESTYLE, alpha=pc.GRID_ALPHA,
                color=pc.GRID_COLOR, linewidth=pc.GRID_LINEWIDTH)

        # Make spines grey
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

        # Only show labels on specific subplots
        if subplot_idx == 0:
            ax.set_ylabel('Macro F1 (\\%)', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        if subplot_idx == 1:
            ax.set_xlabel('Number of Samples (n)', fontsize=pc.AXIS_LABEL_FONT_SIZE)

    # Add shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(METHODS_TO_PLOT),
        fontsize=pc.LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, pc.LEGEND_Y_POSITION)
    )

    # Adjust layout (use slightly wider spacing for scaling plots)
    plt.subplots_adjust(top=pc.SUBPLOT_TOP, bottom=pc.SUBPLOT_BOTTOM,
                       left=pc.SUBPLOT_LEFT, right=pc.SUBPLOT_RIGHT,
                       wspace=0.15)  # Slightly wider than default SUBPLOT_WSPACE (0.1)

    # Save figure
    output_path = pc.FIGURES_DIR / f"scaling_n_{model_name}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)
    print(f"✅ Scaling plot saved to: {output_path}")
    print(f"   Model: {pc.MODEL_NAMES.get(model_name, model_name)}")
    print(f"   Figure size: {pc.ACL_FULL_WIDTH}\" × {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Set publication-quality style (LaTeX rendering)
    pc.set_publication_style()

    print("📊 Generating Self-Consistency Scaling Plots...")
    print(f"Datasets: {DATASETS_TO_PLOT}")
    print(f"Methods: {METHODS_TO_PLOT}")
    print(f"N values: {N_VALUES}")

    # Determine which models to plot
    models_to_plot = [MODEL_TO_PLOT] if MODEL_TO_PLOT else pc.MODEL_ORDER

    # Generate plots for each model
    for model in models_to_plot:
        print(f"\n{'='*80}")
        print(f"Processing model: {pc.MODEL_NAMES.get(model, model)}")
        print(f"{'='*80}")

        # Collect scaling data
        print("\n📁 Loading performance data...")
        scaling_data = collect_scaling_data(model)

        # Create plot
        print("\n🎨 Creating scaling plot...")
        create_scaling_plot(scaling_data, model)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
