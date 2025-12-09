"""
Plot Max Class Probabilities at Reasoning Intervals

Creates publication-ready line plots showing how model confidence evolves during reasoning:
- 3 subplots (D3, DF40, GenImage) side-by-side
- X-axis: Reasoning intervals (0%, 25%, 50%, 75%, 100%)
- Y-axis: Average max class probability (shared scale across subplots)
- 4 lines: Baseline, Prefill (avg CoT+S2), Pseudo-Prefill (avg CoT+S2), Prompt (avg CoT+S2)
- All black lines with different line styles

Data Source: performance_*_with_probs.json files
- Path: intervals["X.XX"]["max_class_probs"]["mean"]

Usage:
    python results/plot_max_class_probs.py --model qwen25-vl-7b
    python results/plot_max_class_probs.py --model llama32-vision-11b
    python results/plot_max_class_probs.py --model llava-onevision-7b

Output: results/figures/max_class_probs_{model}.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared configuration
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset configuration (2k variants)
DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
DATASET_DISPLAY_NAMES = {
    'd3-2k': 'D3',
    'df40-2k': 'DF40',
    'genimage-2k': 'GenImage'
}

# Phrases to average for each mode
PHRASES_TO_AVERAGE = ['cot', 's2']

# Modes to plot (excluding baseline which is separate)
MODES = ['prefill', 'prompt']

# Line configuration: 3 lines, all solid, Paul Tol colorblind-friendly colors
# Using sunset discrete colormap positions 0, 4, 9
LINE_CONFIG = {
    'baseline': {'color': '#364B9A', 'label': 'Baseline'},              # Blue (pos 0)
    'prompt': {'color': '#C2E4EF', 'label': 'Prompt'},                  # Light cyan (pos 4)
    'prefill': {'color': '#DD3D2D', 'label': 'Prefill'}                 # Red (pos 9)
}

LINE_STYLE = '-'  # All solid lines

# Intervals on X-axis
INTERVALS = ['0.00', '0.25', '0.50', '0.75', '1.00']
INTERVAL_LABELS = ['0%', '25%', '50%', '75%', '100%']

# Figure dimensions
FIGURE_HEIGHT = 2.2  # inches

# Subplot spacing (matching radar plot)
SUBPLOT_TOP = 0.92
SUBPLOT_BOTTOM = 0.15
SUBPLOT_LEFT = 0.06
SUBPLOT_RIGHT = 0.98
SUBPLOT_WSPACE = 0.20

# Legend positioning (matching radar plot)
LEGEND_Y_POSITION = 1.15


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_latest_probs_file(dataset: str, model: str, phrase: str,
                           mode: Optional[str] = None) -> Optional[Path]:
    """Find the latest performance_*_with_probs.json file for given configuration."""
    base_dir = Path(__file__).resolve().parent.parent / "output" / dataset / model

    if phrase == 'baseline':
        search_dir = base_dir / 'baseline'
    else:
        if mode is None:
            return None
        search_dir = base_dir / phrase / mode

    if not search_dir.exists():
        return None

    # Find all performance_*_with_probs.json files
    probs_files = list(search_dir.glob('performance_*_with_probs.json'))

    if not probs_files:
        return None

    # Return the most recent file
    return max(probs_files, key=lambda p: p.stat().st_mtime)


def load_interval_data(dataset: str, model: str, phrase: str,
                       mode: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Load max_class_probs mean values for all intervals.

    Returns:
        Dict mapping interval string to mean max_class_prob, or None if not found
    """
    probs_file = find_latest_probs_file(dataset, model, phrase, mode)

    if probs_file is None:
        return None

    try:
        with open(probs_file, 'r') as f:
            data = json.load(f)

        interval_means = {}
        for interval in INTERVALS:
            if interval in data.get('intervals', {}):
                interval_means[interval] = data['intervals'][interval]['max_class_probs']['mean']

        return interval_means if interval_means else None

    except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {probs_file}: {e}")
        return None


def collect_and_average_data(model: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect interval data and compute averages across CoT and S2 for each mode.

    Returns:
        Nested dict: {dataset: {line_key: {interval: mean}}}
        where line_key is 'baseline', 'prefill', 'prefill-pseudo-system', or 'prompt'
    """
    all_data = {}

    for dataset in DATASETS:
        all_data[dataset] = {}

        # Load baseline (no averaging needed)
        baseline_data = load_interval_data(dataset, model, 'baseline', None)
        if baseline_data:
            all_data[dataset]['baseline'] = baseline_data
        else:
            print(f"Warning: No baseline data for {dataset}/{model}")

        # For each mode, average across CoT and S2
        for mode in MODES:
            mode_values = []

            for phrase in PHRASES_TO_AVERAGE:
                interval_data = load_interval_data(dataset, model, phrase, mode)
                if interval_data:
                    mode_values.append(interval_data)
                else:
                    print(f"Warning: No data for {dataset}/{model}/{phrase}/{mode}")

            # Compute average if we have data
            if mode_values:
                averaged = {}
                for interval in INTERVALS:
                    values = [d.get(interval) for d in mode_values if d.get(interval) is not None]
                    if values:
                        averaged[interval] = np.mean(values)
                if averaged:
                    all_data[dataset][mode] = averaged

    return all_data


def get_y_axis_limits(all_data: Dict) -> Tuple[float, float]:
    """
    Compute shared y-axis limits across all subplots.

    Returns:
        Tuple of (y_min, y_max) with some padding
    """
    all_values = []

    for dataset_data in all_data.values():
        for interval_data in dataset_data.values():
            all_values.extend(interval_data.values())

    if not all_values:
        return (0.0, 1.0)

    y_min = min(all_values)
    y_max = max(all_values)

    # Add 5% padding
    padding = (y_max - y_min) * 0.05
    return (y_min - padding, y_max + padding)


# =============================================================================
# PLOTTING
# =============================================================================

def average_across_datasets(all_data: Dict) -> Dict[str, Dict[str, float]]:
    """
    Average interval data across all datasets.

    Args:
        all_data: Nested dict with per-dataset interval data

    Returns:
        Averaged dict: {line_key: {interval: mean}}
    """
    averaged_data = {}

    # Get all line keys across datasets
    all_line_keys = set()
    for dataset_data in all_data.values():
        all_line_keys.update(dataset_data.keys())

    # Average each line across datasets
    for line_key in all_line_keys:
        averaged_data[line_key] = {}

        for interval in INTERVALS:
            values = []
            for dataset_data in all_data.values():
                if line_key in dataset_data and interval in dataset_data[line_key]:
                    values.append(dataset_data[line_key][interval])

            if values:
                averaged_data[line_key][interval] = np.mean(values)

    return averaged_data


def create_max_class_probs_plot(model: str, all_data: Dict):
    """
    Create the 3-subplot figure with max class probability lines.

    Args:
        model: Model name
        all_data: Nested dict with interval data
    """
    # Set publication style
    pc.set_publication_style()

    # Create figure with 3 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 3, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT), sharey=True)

    # Get shared y-axis limits
    y_min, y_max = get_y_axis_limits(all_data)

    # X positions for intervals
    x_positions = np.arange(len(INTERVALS))

    # Line order for plotting (and legend order)
    line_order = ['baseline', 'prompt', 'prefill']

    # Plot each dataset
    for subplot_idx, dataset in enumerate(DATASETS):
        ax = axes[subplot_idx]
        dataset_data = all_data.get(dataset, {})

        # Plot each line
        for line_key in line_order:
            if line_key not in dataset_data:
                continue

            interval_data = dataset_data[line_key]
            config = LINE_CONFIG[line_key]

            # Get y values in interval order
            y_values = [interval_data.get(interval, np.nan) for interval in INTERVALS]

            # Plot line
            ax.plot(x_positions, y_values,
                   color=config['color'], linestyle=LINE_STYLE,
                   linewidth=pc.LINE_WIDTH, marker='o',
                   markersize=pc.MARKER_SIZE)

        # Configure subplot
        ax.set_xlim(-0.3, len(INTERVALS) - 0.7)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(INTERVAL_LABELS, fontsize=pc.TICK_LABEL_FONT_SIZE)
        ax.set_title(DATASET_DISPLAY_NAMES[dataset], fontsize=pc.TITLE_FONT_SIZE)

        # Y-axis label only on leftmost subplot
        if subplot_idx == 0:
            ax.set_ylabel('Class Confidence', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        # X-axis label only on middle subplot
        if subplot_idx == 1:
            ax.set_xlabel('Reasoning Interval', fontsize=pc.AXIS_LABEL_FONT_SIZE)

        # Grid
        ax.grid(True, color=pc.GRID_COLOR, linestyle=pc.GRID_LINESTYLE,
               linewidth=pc.GRID_LINEWIDTH, alpha=pc.GRID_ALPHA)
        ax.set_axisbelow(True)

        # Tick label size
        ax.tick_params(axis='both', labelsize=pc.TICK_LABEL_FONT_SIZE)

    # Create legend with 4 grayscale colors
    legend_handles = [
        mlines.Line2D([], [], color=LINE_CONFIG[key]['color'], linestyle=LINE_STYLE,
                     linewidth=pc.LINE_WIDTH, label=LINE_CONFIG[key]['label'])
        for key in line_order
    ]

    # Add legend at top center
    fig.legend(handles=legend_handles, loc='upper center',
              ncol=len(legend_handles), fontsize=pc.LEGEND_FONT_SIZE,
              frameon=False, bbox_to_anchor=(0.5, LEGEND_Y_POSITION))

    # Adjust layout
    plt.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                       left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
                       wspace=SUBPLOT_WSPACE)

    # Save figure
    model_short = pc.MODEL_NAMES.get(model, model)
    output_path = pc.FIGURES_DIR / f"max_class_probs_{model_short.lower()}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)

    print(f"Saved: {output_path}")
    print(f"   Model: {model_short}")
    print(f"   Figure size: {pc.ACL_FULL_WIDTH}\" x {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI")

    plt.close()


def create_averaged_max_class_probs_plot(model: str, all_data: Dict):
    """
    Create a single-plot figure with averaged max class probability across all datasets.

    Args:
        model: Model name
        all_data: Nested dict with per-dataset interval data
    """
    # Set publication style
    pc.set_publication_style()

    # Average data across all datasets
    averaged_data = average_across_datasets(all_data)

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(3.0, FIGURE_HEIGHT))

    # Get y-axis limits
    all_values = [v for interval_data in averaged_data.values() for v in interval_data.values()]
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = 0.0, 1.0

    # X positions for intervals
    x_positions = np.arange(len(INTERVALS))

    # Line order for plotting (and legend order)
    line_order = ['baseline', 'prompt', 'prefill']

    # Plot each line
    for line_key in line_order:
        if line_key not in averaged_data:
            continue

        interval_data = averaged_data[line_key]
        config = LINE_CONFIG[line_key]

        # Get y values in interval order
        y_values = [interval_data.get(interval, np.nan) for interval in INTERVALS]

        # Plot line
        ax.plot(x_positions, y_values,
               color=config['color'], linestyle=LINE_STYLE,
               linewidth=pc.LINE_WIDTH, marker='o',
               markersize=pc.MARKER_SIZE)

    # Configure subplot
    ax.set_xlim(-0.3, len(INTERVALS) - 0.7)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(INTERVAL_LABELS, fontsize=pc.TICK_LABEL_FONT_SIZE)
    ax.set_ylabel('Class Confidence', fontsize=pc.AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel('Reasoning Interval', fontsize=pc.AXIS_LABEL_FONT_SIZE)

    # Grid
    ax.grid(True, color=pc.GRID_COLOR, linestyle=pc.GRID_LINESTYLE,
           linewidth=pc.GRID_LINEWIDTH, alpha=pc.GRID_ALPHA)
    ax.set_axisbelow(True)

    # Tick label size
    ax.tick_params(axis='both', labelsize=pc.TICK_LABEL_FONT_SIZE)

    # Create legend
    legend_handles = [
        mlines.Line2D([], [], color=LINE_CONFIG[key]['color'], linestyle=LINE_STYLE,
                     linewidth=pc.LINE_WIDTH, label=LINE_CONFIG[key]['label'])
        for key in line_order
    ]

    # Add legend at top center (where title would be)
    fig.legend(handles=legend_handles, loc='upper center',
              ncol=len(legend_handles), fontsize=pc.LEGEND_FONT_SIZE,
              frameon=False, bbox_to_anchor=(0.5, 1.02))

    # Adjust layout with space for legend at top
    plt.subplots_adjust(top=0.90, bottom=SUBPLOT_BOTTOM, left=0.12, right=0.95)

    # Save figure
    model_short = pc.MODEL_NAMES.get(model, model)
    output_path = pc.FIGURES_DIR / f"max_class_probs_{model_short.lower()}_avg.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)

    print(f"Saved: {output_path}")
    print(f"   Model: {model_short}")
    print(f"   Figure size: 3.0\" x {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot max class probabilities at reasoning intervals"
    )
    parser.add_argument(
        '--model', '-m', required=True,
        choices=pc.MODEL_ORDER,
        help='Model to generate plot for'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print(f"Generating max class probability plots for {args.model}...")
    print(f"Datasets: {', '.join(DATASETS)}")

    # Collect and average data
    print("Loading interval data and averaging CoT+S2 for each mode...")
    all_data = collect_and_average_data(args.model)

    # Check if we have any data
    total_entries = sum(len(d) for d in all_data.values())
    if total_entries == 0:
        print("Error: No data found. Run add_class_probs_to_reasoning.py first.")
        sys.exit(1)

    print(f"Found data for {total_entries} line configurations across datasets")

    # Create per-dataset plot
    print("Creating per-dataset plot...")
    create_max_class_probs_plot(args.model, all_data)

    # Create averaged plot
    print("Creating averaged plot...")
    create_averaged_max_class_probs_plot(args.model, all_data)

    print("Done!")


if __name__ == '__main__':
    main()
