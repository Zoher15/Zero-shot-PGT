"""
Recall Radar Plot Generator

Creates publication-ready radar plots comparing recall performance across AI generators:
- 3 datasets (D3, DF40, GenImage) as 3 side-by-side radar plots
- Single model view (configurable)
- 3 methods (baseline, cot, s2) as colored lines with shaded regions

Features:
- Generator-level recall analysis (excludes 'real' images)
- Transparent shaded regions for each method
- Colorblind-friendly palette (matching bar chart)
- Shared legend across subplots
- 0-100% scale for all radar plots

Output: recall_radar_{model}.pdf in project root
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from math import pi

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import config and shared plotting config
import config
import helpers
import plot_config as pc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Generate plots for all models (set to None to plot all, or specify a single model)
MODEL_TO_PLOT = None  # None = all models, or specify: 'llava-onevision-7b', 'qwen25-vl-7b', 'llama32-vision-11b'

# Figure dimensions (ACL format)
FIGURE_HEIGHT = 2.5  # inches (adjusted for radar plots)

# Font size overrides for radar plots (smaller due to crowded space)
RADAR_SPOKE_LABEL_SIZE = 6   # Generator labels on spokes (need smaller)
RADAR_TICK_SIZE = 5          # Radial tick labels (0%, 25%, etc.)

# Evaluation parameters
PHRASE_MODE = pc.DEFAULT_PHRASE_MODE  # Which phrase mode to load
N_RESPONSES = pc.DEFAULT_N_RESPONSES  # Which n value to load

# Phrase mapping (internal phrase names to methods)
PHRASE_TO_METHOD = {
    'baseline': 'baseline',
    'cot': 'cot',
    's2': 's2'
}

# Radar plot configuration
RADIAL_TICKS = [0, 25, 50, 75, 100]  # Y-axis ticks (recall percentages)

# Subplot spacing override for radar plots
SUBPLOT_TOP = 0.92
SUBPLOT_BOTTOM = 0.12
SUBPLOT_LEFT = 0.05
SUBPLOT_RIGHT = 0.98
SUBPLOT_WSPACE_RADAR = 0.25  # Wider spacing for radar plots
LEGEND_Y_POSITION_RADAR = 1.05  # Different from bar plots

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_generator_from_path(image_path: str, dataset: str) -> str:
    """
    Extract generator class from image path.

    Args:
        image_path: Full path to image
        dataset: Dataset name (d3, df40, genimage)

    Returns:
        Generator class string
    """
    from pathlib import Path

    if dataset == 'd3':
        # D3: Generator in filename (e.g., "441_image_gen2.png" -> "image_gen2")
        filename = Path(image_path).stem  # Remove extension
        if '_' in filename:
            parts = filename.split('_')
            # Last part is generator (image_gen0, image_gen1, etc.)
            if len(parts) >= 2:
                return '_'.join(parts[1:])  # Join all parts after first underscore
        return 'unknown'

    elif dataset == 'df40':
        # DF40: Generator in directory path (e.g., "/DF40/MidJourney/..." -> "midjourney")
        path_parts = Path(image_path).parts
        # Find DF40 index and get next directory
        for i, part in enumerate(path_parts):
            if 'DF40' in part and i + 1 < len(path_parts):
                generator = path_parts[i + 1].lower()
                return generator
        return 'unknown'

    elif dataset == 'genimage':
        # GenImage: Generator in directory path (e.g., "/genimage/stable_diffusion_v_1_5/..." -> "stable_diffusion_v_1_5")
        path_parts = Path(image_path).parts
        # Find genimage index and get next directory
        for i, part in enumerate(path_parts):
            if 'genimage' in part.lower() and i + 1 < len(path_parts):
                generator = path_parts[i + 1]
                return generator
        return 'unknown'

    return 'unknown'


def load_reasoning_json_with_subsets(dataset: str, model: str, phrase: str,
                                     mode: str = 'prefill', n: int = 1) -> Dict[str, Dict[str, int]]:
    """
    Load reasoning JSON and compute TP/FN counts per generator subset.

    Args:
        dataset: Dataset name
        model: Model name
        phrase: Phrase name
        mode: Phrase mode
        n: Number of responses

    Returns:
        Dict mapping subset -> {TP: count, FN: count}
    """
    # Load reasoning data
    reasoning_data = helpers.load_reasoning_json(dataset, model, phrase, mode, n)

    # Count TP/FN per subset
    subset_counts = defaultdict(lambda: {'TP': 0, 'FN': 0})

    for result in reasoning_data:
        image_path = result['image']
        ground_truth = result['ground_truth']
        prediction = result['aggregated_prediction']

        # Skip real images
        if ground_truth == 'real':
            continue

        # Extract generator from path
        subset = extract_generator_from_path(image_path, dataset)

        # Count TP/FN for AI-generated images
        if prediction == 'ai-generated' and ground_truth == 'ai-generated':
            subset_counts[subset]['TP'] += 1
        elif prediction == 'real' and ground_truth == 'ai-generated':
            subset_counts[subset]['FN'] += 1

    return dict(subset_counts)


def compute_recall_per_subset(subset_counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Compute recall percentage for each subset.

    Args:
        subset_counts: Dict mapping subset -> {TP: count, FN: count}

    Returns:
        Dict mapping subset -> recall percentage (0-100)
    """
    recall_scores = {}

    for subset, counts in subset_counts.items():
        tp = counts['TP']
        fn = counts['FN']

        # Calculate recall: TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        recall_scores[subset] = recall * 100  # Convert to percentage

    return recall_scores


def collect_recall_data_for_model(model: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect recall data for all datasets and methods for a given model.

    Args:
        model: Model name

    Returns:
        Nested dict: {dataset: {method: {subset: recall_percentage}}}
    """
    results = {}

    for dataset in pc.DATASET_ORDER:
        results[dataset] = {}

        for method, phrase in PHRASE_TO_METHOD.items():
            try:
                subset_counts = load_reasoning_json_with_subsets(
                    dataset, model, phrase, PHRASE_MODE, N_RESPONSES
                )
                recall_scores = compute_recall_per_subset(subset_counts)

                # Filter out 'real' and 'unknown' subsets
                filtered_scores = {k: v for k, v in recall_scores.items()
                                  if k not in ['real', 'unknown']}

                results[dataset][method] = filtered_scores

            except Exception as e:
                print(f"Warning: Could not load data for {dataset}/{model}/{phrase}: {e}")
                results[dataset][method] = {}

    return results


def create_radar_plot(ax, data: Dict[str, List[float]], labels: List[str],
                     title: str, show_ylabel: bool = False):
    """
    Create a single radar plot.

    Args:
        ax: Matplotlib axis
        data: Dict mapping method -> list of recall values
        labels: List of generator labels (for spokes)
        title: Subplot title
        show_ylabel: Whether to show y-axis label
    """
    num_vars = len(labels)

    if num_vars == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, -0.15, title, ha='center', va='top',
               transform=ax.transAxes, fontsize=pc.TITLE_FONT_SIZE)
        return

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle

    # Initialize polar plot
    ax = plt.subplot(ax.get_subplotspec(), polar=True)

    # Draw one axis per variable and add labels
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=RADAR_SPOKE_LABEL_SIZE)

    # Find max value across all methods for this dataset
    max_value = 0
    for method_values in data.values():
        if method_values:
            max_value = max(max_value, max(method_values))

    # Round max_value up to nearest 20
    y_max = np.ceil(max_value / 20) * 20
    if y_max == 0:
        y_max = 100  # Fallback

    # Generate ticks at every 20% interval
    radial_ticks = list(range(0, int(y_max) + 1, 20))

    # Set y-axis (radial) limits and ticks
    ax.set_ylim(0, y_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{int(t)}%" for t in radial_ticks], fontsize=RADAR_TICK_SIZE)
    ax.yaxis.set_tick_params(pad=0.5)

    # Grid
    ax.grid(color=pc.GRID_COLOR, linestyle=pc.GRID_LINESTYLE, linewidth=pc.GRID_LINEWIDTH, alpha=pc.GRID_ALPHA)

    # Plot data for each method
    for method in pc.METHOD_ORDER:
        if method not in data or not data[method]:
            continue

        values = data[method]
        values += values[:1]  # Complete the circle

        # Plot line
        ax.plot(angles, values, 'o-', linewidth=pc.LINE_WIDTH,
               color=pc.METHOD_COLORS[method], label=pc.METHOD_NAMES[method],
               markersize=pc.MARKER_SIZE)

        # Fill area
        ax.fill(angles, values, alpha=pc.FILL_ALPHA, color=pc.METHOD_COLORS[method])

    # Title at bottom instead of top
    ax.text(0.5, -0.20, title, ha='center', va='top',
           transform=ax.transAxes, fontsize=pc.TITLE_FONT_SIZE)


def create_recall_radar_plots(model: str, recall_data: Dict):
    """
    Create 3 radar plots (one per dataset) for a given model.

    Args:
        model: Model name
        recall_data: Nested dict {dataset: {method: {subset: recall}}}
    """
    fig = plt.figure(figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT))

    # Track S2 improvements across all datasets/generators
    s2_improvements = []  # List of (improvement, dataset, generator) tuples

    # Create 3 subplots
    axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='polar')
        axes.append(ax)

    for subplot_idx, dataset in enumerate(pc.DATASET_ORDER):
        ax = axes[subplot_idx]
        dataset_data = recall_data.get(dataset, {})

        # Get all unique subsets across methods
        all_subsets = set()
        for method_data in dataset_data.values():
            all_subsets.update(method_data.keys())

        # Sort subsets for consistent ordering
        sorted_subsets = sorted(list(all_subsets))

        # Map subsets to display names
        subset_labels = [pc.GENERATOR_NAMES.get(s, s.upper()).lower() for s in sorted_subsets]

        # Prepare data for radar plot
        plot_data = {}
        for method in pc.METHOD_ORDER:
            method_scores = dataset_data.get(method, {})
            # Ensure all subsets have values (use 0 if missing)
            values = [method_scores.get(s, 0) for s in sorted_subsets]
            plot_data[method] = values

        # Compute S2 improvements for this dataset
        for generator in sorted_subsets:
            baseline_recall = dataset_data.get('baseline', {}).get(generator, 0)
            cot_recall = dataset_data.get('cot', {}).get(generator, 0)
            s2_recall = dataset_data.get('s2', {}).get(generator, 0)

            next_best = max(baseline_recall, cot_recall)
            if next_best > 0:
                rel_improvement = ((s2_recall - next_best) / next_best) * 100
                s2_improvements.append((rel_improvement, dataset, generator))

        # Create radar plot
        title = pc.DATASET_NAMES[dataset]
        create_radar_plot(ax, plot_data, subset_labels, title,
                         show_ylabel=(subplot_idx == 0))

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(pc.METHOD_ORDER),
                  fontsize=pc.LEGEND_FONT_SIZE, frameon=False,
                  bbox_to_anchor=(0.5, LEGEND_Y_POSITION_RADAR))

    # Adjust layout
    plt.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                       left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
                       wspace=SUBPLOT_WSPACE_RADAR)

    # Save figure
    model_short = pc.MODEL_NAMES.get(model, model)
    output_path = pc.FIGURES_DIR / f"recall_radar_{model_short.lower()}.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)
    print(f"✅ Radar plot saved to: {output_path}")
    print(f"   Model: {model_short}")
    print(f"   Figure size: {pc.ACL_FULL_WIDTH}\" × {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI (ACL format)")

    # Print S2 improvement statistics
    if s2_improvements:
        min_improvement = min(s2_improvements, key=lambda x: x[0])
        max_improvement = max(s2_improvements, key=lambda x: x[0])
        mean_value = np.mean([x[0] for x in s2_improvements])

        print(f"\n📈 S2 Relative Improvement over next best (baseline/CoT):")
        print(f"   Min: {min_improvement[0]:+.2f}% ({pc.DATASET_NAMES[min_improvement[1]]}, {pc.GENERATOR_NAMES.get(min_improvement[2], min_improvement[2])})")
        print(f"   Max: {max_improvement[0]:+.2f}% ({pc.DATASET_NAMES[max_improvement[1]]}, {pc.GENERATOR_NAMES.get(max_improvement[2], max_improvement[2])})")
        print(f"   Mean: {mean_value:+.2f}%")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Set publication-quality style (LaTeX rendering)
    pc.set_publication_style()

    print("📊 Generating Recall Radar Plots...")
    print(f"Configuration: mode={PHRASE_MODE}, n={N_RESPONSES}")

    # Determine which models to plot
    models_to_plot = [MODEL_TO_PLOT] if MODEL_TO_PLOT else pc.MODEL_ORDER

    # Generate plots for each model
    for model in models_to_plot:
        print(f"\n{'='*80}")
        print(f"Processing model: {pc.MODEL_NAMES.get(model, model)}")
        print(f"{'='*80}")

        # Collect recall data
        print("📁 Loading reasoning data and computing recall per generator...")
        recall_data = collect_recall_data_for_model(model)

        # Create plots
        print("🎨 Creating radar plots...")
        create_recall_radar_plots(model, recall_data)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
