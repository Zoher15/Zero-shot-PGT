"""
Macro F1 Bar Plot Generator

Creates a publication-ready bar plot comparing macro F1-scores across:
- 3 datasets (D3, DF40, GenImage)
- 3 models (LLaVa, Qwen, Llama)
- 3 methods (baseline, cot, s2)

Features:
- 95% confidence intervals (optional, loaded from confidence.json)
- Relative improvement markers (s2 vs cot)
- Colorblind-friendly palette
- Shared legend across subplots
- Automatic y-axis scaling (min/max rounded to nearest 10)

Output: macro_f1_bars.pdf in project root
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared config and project config
import config
import plot_config as pc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Figure dimensions (ACL format)
FIGURE_HEIGHT = 2.25  # inches (adjusted aspect ratio for ACL)

# Evaluation parameters
PHRASE_MODE = pc.DEFAULT_PHRASE_MODE  # Which phrase mode to load
N_RESPONSES = pc.DEFAULT_N_RESPONSES  # Which n value to load

# Confidence interval toggle
SHOW_CI = True  # Set to False to hide error bars

# Output configuration
OUTPUT_PATH = pc.FIGURES_DIR / "macro_f1_bars.pdf"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_performance_json(dataset: str, model: str, phrase: str, mode: str = 'prefill', n: int = 1) -> Optional[Dict[str, Any]]:
    """
    Load performance JSON for given configuration.

    Args:
        dataset: Dataset name (e.g., 'd3', 'df40')
        model: Model name (e.g., 'qwen25-vl-7b')
        phrase: Phrase name (e.g., 'baseline', 'cot', 's2')
        mode: Phrase mode (default: 'prefill')
        n: Number of responses (default: 1)

    Returns:
        Performance data dictionary or None if not found
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)

    # Find most recent performance_fixed JSON (with null handling)
    performance_files = sorted(output_dir.glob("performance_fixed_*.json"))
    if not performance_files:
        print(f"⚠️  No performance JSON found for {dataset}/{model}/{phrase}")
        return None

    latest_file = performance_files[-1]  # Most recent timestamp

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return data


def load_confidence_json(dataset: str, model: str, phrase: str, mode: str = 'prefill', n: int = 1) -> Optional[Dict[str, Any]]:
    """
    Load confidence interval JSON for given configuration.

    Args:
        dataset: Dataset name
        model: Model name
        phrase: Phrase name
        mode: Phrase mode (default: 'prefill')
        n: Number of responses (default: 1)

    Returns:
        Confidence data dictionary or None if not found
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)
    confidence_file = output_dir / "confidence.json"

    if not confidence_file.exists():
        return None

    with open(confidence_file, 'r') as f:
        data = json.load(f)

    return data


def collect_all_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect macro F1 scores for all dataset/model/method combinations.

    Returns:
        Nested dict: {dataset: {model: {method: f1_score}}}
    """
    results = {}

    for dataset in pc.DATASET_ORDER:
        results[dataset] = {}

        for model in pc.MODEL_ORDER:
            results[dataset][model] = {}

            for method in pc.METHOD_ORDER:
                perf_data = load_performance_json(dataset, model, method, PHRASE_MODE, N_RESPONSES)

                if perf_data:
                    macro_f1 = perf_data['metrics']['macro_f1']
                    results[dataset][model][method] = macro_f1 * 100  # Convert to percentage
                else:
                    results[dataset][model][method] = None

    return results


def collect_all_confidence_intervals() -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    Collect 95% CI for macro F1 for all dataset/model/method combinations.

    Returns:
        Nested dict: {dataset: {model: {method: (lower, upper)}}}
    """
    ci_data = {}

    for dataset in pc.DATASET_ORDER:
        ci_data[dataset] = {}

        for model in pc.MODEL_ORDER:
            ci_data[dataset][model] = {}

            for method in pc.METHOD_ORDER:
                conf_data = load_confidence_json(dataset, model, method, PHRASE_MODE, N_RESPONSES)

                if conf_data and 'macro_f1' in conf_data:
                    ci = conf_data['macro_f1']['ci_95']
                    ci_data[dataset][model][method] = (ci[0] * 100, ci[1] * 100)  # Convert to percentage
                else:
                    ci_data[dataset][model][method] = None

    return ci_data


def compute_global_y_limits(results: Dict) -> tuple:
    """
    Compute global y-axis limits across all subplots.
    Round to nearest 5 for clean axes.

    Args:
        results: Nested dict of F1 scores

    Returns:
        (y_min, y_max) tuple
    """
    all_values = []

    for dataset_data in results.values():
        for model_data in dataset_data.values():
            for f1_score in model_data.values():
                if f1_score is not None:
                    all_values.append(f1_score)

    if not all_values:
        return (0, 100)

    min_val = min(all_values)
    max_val = max(all_values)

    # Round to nearest 5
    y_min = np.floor(min_val / 5) * 5
    y_max = np.ceil(max_val / 5) * 5

    return (y_min, y_max)


def create_macro_f1_bar_plot(results: Dict, ci_data: Optional[Dict] = None):
    """
    Create 3-subplot bar plot comparing macro F1 across datasets/models/methods.

    Args:
        results: Nested dict {dataset: {model: {method: f1_score}}}
        ci_data: Optional nested dict {dataset: {model: {method: (lower, upper)}}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT))

    # Bar width and positions
    n_models = len(pc.MODEL_ORDER)
    n_methods = len(pc.METHOD_ORDER)
    bar_width = pc.BAR_WIDTH
    group_spacing = pc.GROUP_SPACING

    for subplot_idx, dataset in enumerate(pc.DATASET_ORDER):
        ax = axes[subplot_idx]

        # Set subplot title (lower pad to make room for legend)
        ax.set_title(pc.DATASET_NAMES[dataset], fontsize=pc.TITLE_FONT_SIZE, pad=5)

        # Prepare data for this dataset
        dataset_results = results[dataset]

        # Compute y-axis limits for this dataset only
        dataset_values = []
        for model_data in dataset_results.values():
            for f1_score in model_data.values():
                if f1_score is not None:
                    dataset_values.append(f1_score)

        if dataset_values:
            max_val = max(dataset_values)
            y_min = 0  # Fixed at 0 for all subplots
            y_max = np.ceil(max_val / 5) * 5
        else:
            y_min, y_max = 0, 100

        # X positions for model groups
        x_positions = np.arange(n_models) * group_spacing

        # Plot bars for each method
        for method_idx, method in enumerate(pc.METHOD_ORDER):
            offset = (method_idx - 1) * bar_width  # Center the 3 bars around group center

            # Collect F1 scores and CIs for this method across models
            f1_scores = [dataset_results[model][method] for model in pc.MODEL_ORDER]

            # Prepare error bars if CI data available
            yerr = None
            if SHOW_CI and ci_data:
                ci_lower = []
                ci_upper = []

                for model in pc.MODEL_ORDER:
                    ci = ci_data[dataset][model][method]
                    if ci:
                        f1 = dataset_results[model][method]
                        ci_lower.append(f1 - ci[0])
                        ci_upper.append(ci[1] - f1)
                    else:
                        ci_lower.append(0)
                        ci_upper.append(0)

                yerr = [ci_lower, ci_upper]

            # Plot bars with black borders
            ax.bar(
                x_positions + offset,
                f1_scores,
                bar_width,
                label=pc.METHOD_NAMES[method],
                color=pc.METHOD_COLORS[method],
                edgecolor='black',
                linewidth=pc.BAR_EDGE_WIDTH,
                yerr=yerr,
                capsize=pc.ERROR_BAR_CAPSIZE,
                error_kw={'linewidth': pc.ERROR_BAR_WIDTH}
            )

            # Add improvement markers for s2
            if method == 's2':
                for model_idx, model in enumerate(pc.MODEL_ORDER):
                    s2_f1 = dataset_results[model]['s2']
                    cot_f1 = dataset_results[model]['cot']
                    baseline_f1 = dataset_results[model]['baseline']

                    if s2_f1 is not None:
                        # Find the next best method (CoT or baseline)
                        best_other_f1 = None
                        if cot_f1 is not None and baseline_f1 is not None:
                            best_other_f1 = max(cot_f1, baseline_f1)
                        elif cot_f1 is not None:
                            best_other_f1 = cot_f1
                        elif baseline_f1 is not None:
                            best_other_f1 = baseline_f1

                        if best_other_f1 is not None and best_other_f1 > 0:
                            improvement = ((s2_f1 - best_other_f1) / best_other_f1) * 100

                            # Position above bar (with CI if available)
                            y_pos = s2_f1
                            if SHOW_CI and ci_data and ci_data[dataset][model]['s2']:
                                y_pos = ci_data[dataset][model]['s2'][1]  # Upper CI

                            ax.text(
                                x_positions[model_idx] + offset,
                                y_pos + 0.5,  # Lower position (was 1)
                                f'{improvement:+.0f}%',  # Use :+ to auto-add sign (+ or -)
                                ha='center',
                                va='bottom',
                                fontsize=pc.ANNOTATION_FONT_SIZE,
                                color='black',
                                fontweight='normal'
                            )

        # Configure x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels([pc.MODEL_NAMES[m] for m in pc.MODEL_ORDER], fontsize=pc.TICK_LABEL_FONT_SIZE)

        # Configure y-axis
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
        ax.grid(axis='y', linestyle='-', alpha=pc.GRID_ALPHA, color=pc.GRID_COLOR)

        # Make spines (box borders) grey
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

        # Only show y-label on leftmost subplot
        if subplot_idx == 0:
            ax.set_ylabel('Macro F1 (\\%)', fontsize=pc.AXIS_LABEL_FONT_SIZE)

    # Add shared legend at top (higher position since dataset labels moved down)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=n_methods,
        fontsize=pc.LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, pc.LEGEND_Y_POSITION)
    )

    # Adjust layout with precise control over spacing
    plt.subplots_adjust(top=pc.SUBPLOT_TOP, bottom=pc.SUBPLOT_BOTTOM, left=pc.SUBPLOT_LEFT,
                       right=pc.SUBPLOT_RIGHT, wspace=pc.SUBPLOT_WSPACE)

    # Save figure (300 DPI for publication quality)
    plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)
    print(f"✅ Plot saved to: {OUTPUT_PATH}")
    print(f"   Figure size: {pc.ACL_FULL_WIDTH}\" × {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI (ACL format)")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Set publication-quality style (LaTeX rendering)
    pc.set_publication_style()

    print("📊 Generating Macro F1 Bar Plot...")
    print(f"Configuration: mode={PHRASE_MODE}, n={N_RESPONSES}")

    # Collect all results
    print("\n📁 Loading performance data...")
    results = collect_all_results()

    # Collect confidence intervals if enabled
    ci_data = None
    if SHOW_CI:
        print("📁 Loading confidence intervals...")
        ci_data = collect_all_confidence_intervals()

    # Create plot
    print("\n🎨 Creating plot...")
    create_macro_f1_bar_plot(results, ci_data)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
