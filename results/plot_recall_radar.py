"""
Recall Bar Plot Generator

Creates publication-ready grouped bar plots comparing recall performance across AI generators:
- 3 datasets (D3, DF40, GenImage) as 3 side-by-side subplots
- Single model view (configurable)
- 5 method-mode combinations as grouped vertical bars per generator

Features:
- Generator-level recall analysis (excludes 'real' images)
- Color encodes method (baseline/CoT/S2), hatching encodes mode (solid=prefill, hatched=prompt)
- Colorblind-friendly palette
- Shared legend across subplots

Output: recall_bars_{model}.pdf in figures/
"""

import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import helpers
import plot_config as pc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Generate plots for all models (set to None to plot all, or specify a single model)
MODEL_TO_PLOT = None  # None = all models, or specify a single model

# Figure dimensions
FIGURE_HEIGHT = 2.6  # inches

# Evaluation parameters
N_RESPONSES = pc.DEFAULT_N_RESPONSES

# 7 bar slots per generator: (phrase, mode)
BAR_SLOTS = [
    ('baseline', 'prefill'),                # single baseline bar (mode ignored)
    ('cot',      'prompt'),
    ('cot',      'prefill-pseudo-system'),
    ('cot',      'prefill'),
    ('s2',       'prompt'),
    ('s2',       'prefill-pseudo-system'),
    ('s2',       'prefill'),
]

# Bar width for 5 bars per generator group
BAR_WIDTH = 0.17

# Output path
OUTPUT_PATH = pc.FIGURES_DIR / "recall_bars"

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


def collect_recall_data_for_model(model: str) -> Dict[str, Dict]:
    """
    Collect recall data for all datasets and bar slots for a given model.

    Args:
        model: Model name

    Returns:
        Nested dict: {dataset: {(method,mode): {subset: recall_pct}, '_generators': [sorted list]}}
    """
    results = {}

    for dataset in pc.DATASET_ORDER:
        results[dataset] = {}
        all_subsets = set()

        for method, mode in BAR_SLOTS:
            try:
                subset_counts = load_reasoning_json_with_subsets(
                    dataset, model, method, mode, N_RESPONSES
                )
                recall_scores = compute_recall_per_subset(subset_counts)
                filtered = {k: v for k, v in recall_scores.items()
                           if k not in ['real', 'unknown']}
                results[dataset][(method, mode)] = filtered
                all_subsets.update(filtered.keys())
            except Exception as e:
                print(f"Warning: Could not load {dataset}/{model}/{method}/{mode}: {e}")
                results[dataset][(method, mode)] = {}

        results[dataset]['_generators'] = sorted(list(all_subsets))

    return results


def create_recall_bar_plots(model: str, recall_data: Dict):
    """
    Create a single-axis bar plot with all generators on one x-axis,
    separated by vertical lines and dataset labels.

    Bar layout per generator:
      baseline | cot-prompt cot-prefill | s2-prompt s2-prefill
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(pc.ACL_FULL_WIDTH, FIGURE_HEIGHT))

    n_bars = len(BAR_SLOTS)  # 5
    bar_width = BAR_WIDTH
    dataset_gap = 0.4  # extra gap between dataset groups

    # Build a flat list of generators across all datasets, tracking boundaries
    all_gen_keys = []      # raw generator keys
    all_gen_labels = []    # display labels
    all_gen_datasets = []  # which dataset each belongs to
    dataset_boundaries = []  # x positions of boundaries between datasets
    x_positions = []
    current_x = 0

    # Generator spacing: 5 bars + 1 bar gap between groups
    gen_spacing = n_bars * bar_width + dataset_gap

    for ds_idx, dataset in enumerate(pc.DATASET_ORDER):
        generators = recall_data.get(dataset, {}).get('_generators', [])

        if ds_idx > 0 and generators:
            current_x += dataset_gap  # gap between datasets

        boundary_start = current_x

        for gen in generators:
            x_positions.append(current_x)
            all_gen_keys.append((dataset, gen))
            all_gen_labels.append(pc.GENERATOR_NAMES.get(gen, gen))
            all_gen_datasets.append(dataset)
            current_x += gen_spacing

        if generators:
            last_gen_x = current_x - gen_spacing  # position of last generator
            dataset_boundaries.append({
                'dataset': dataset,
                'start': boundary_start - gen_spacing / 2,
                'end': last_gen_x + gen_spacing / 2,
                'center': (boundary_start + last_gen_x) / 2,
            })

    x_positions = np.array(x_positions)

    # Plot bars
    for bar_idx, (method, mode) in enumerate(BAR_SLOTS):
        offset = (bar_idx - (n_bars - 1) / 2) * bar_width

        values = []
        for dataset, gen in all_gen_keys:
            slot_data = recall_data.get(dataset, {}).get((method, mode), {})
            values.append(slot_data.get(gen, 0))

        alpha = pc.MODE_ALPHAS[mode] if method != 'baseline' else 1.0
        import matplotlib.colors as mcolors
        rgba = list(mcolors.to_rgba(pc.METHOD_COLORS[method]))
        rgba[3] = alpha

        ax.bar(
            x_positions + offset,
            values,
            bar_width,
            color=rgba,
            edgecolor='black',
            linewidth=pc.BAR_EDGE_WIDTH,
        )

    # X-axis: generator labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_gen_labels, fontsize=pc.TICK_LABEL_FONT_SIZE,
                       rotation=45, ha='right')

    # Set xlim so outer padding matches the separator-to-bar-edge gap
    # Separator is at (1 + dataset_gap) / 2 from each adjacent bar center
    # so outer edge should be the same distance from the first/last bar center
    edge_pad = (gen_spacing + dataset_gap) / 2
    ax.set_xlim(x_positions[0] - edge_pad, x_positions[-1] + edge_pad)

    # Dataset separator lines (only between datasets, not at edges) and labels
    for i, boundary in enumerate(dataset_boundaries):
        # Vertical separator line only between datasets (not before first or after last)
        if i > 0:
            midpoint = (dataset_boundaries[i - 1]['end'] + boundary['start']) / 2
            ax.axvline(x=midpoint, color='gray', linewidth=0.8,
                      linestyle='-', alpha=0.5, zorder=0)

        # Dataset label at top of plot area
        ax.text(boundary['center'], 1.02, pc.DATASET_NAMES[boundary['dataset']],
               ha='center', va='bottom', fontsize=pc.TITLE_FONT_SIZE,
               transform=ax.get_xaxis_transform())

    # Y-axis
    all_values = [v for ds_data in recall_data.values()
                 for slot in BAR_SLOTS
                 for v in ds_data.get(slot, {}).values()]
    y_max = np.ceil(max(all_values) / 10) * 10 if all_values else 100
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Recall (\\%)', fontsize=pc.AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', linestyle='-', alpha=pc.GRID_ALPHA, color=pc.GRID_COLOR)

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    # Legend: color patches for methods + mode circles
    from matplotlib.lines import Line2D

    method_handles = [
        mpatches.Patch(facecolor=pc.METHOD_COLORS[m], edgecolor='black',
                       linewidth=0.5, label=pc.METHOD_NAMES[m])
        for m in pc.METHOD_ORDER if m != 'baseline'
    ]
    none_handle = Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                         markeredgecolor='black', markeredgewidth=0.5,
                         markersize=7, label='None')
    mode_handles = []
    for mode in ['prompt', 'prefill-pseudo-system', 'prefill']:
        rgba = list(mcolors.to_rgba('gray'))
        rgba[3] = pc.MODE_ALPHAS[mode]
        mode_handles.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=rgba,
                   markeredgecolor='black', markeredgewidth=0.5,
                   markersize=7, label=pc.MODE_NAMES[mode])
        )
    all_handles = method_handles + [none_handle] + mode_handles

    plt.subplots_adjust(top=0.87, bottom=0.13, left=0.075, right=0.99)

    fig.legend(
        handles=all_handles,
        loc='upper center',
        ncol=len(all_handles),
        fontsize=pc.LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    # Save
    model_short = pc.MODEL_NAMES.get(model, model)
    output_base = pc.FIGURES_DIR / f"recall_bars_{model_short.lower()}"
    pdf_path, png_path = pc.save_figure(fig, output_base)
    print(f"Recall bar plots saved to:")
    print(f"   PDF: {pdf_path}")
    print(f"   PNG: {png_path}")
    print(f"   Model: {model_short}")
    print(f"   Figure size: {pc.ACL_FULL_WIDTH}\" x {FIGURE_HEIGHT}\" @ {pc.PUBLICATION_DPI} DPI")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Set publication-quality style (LaTeX rendering)
    pc.set_publication_style()

    print("Generating Recall Bar Plots...")
    print(f"Configuration: n={N_RESPONSES}")

    # Determine which models to plot
    models_to_plot = [MODEL_TO_PLOT] if MODEL_TO_PLOT else pc.MODEL_ORDER

    # Generate plots for each model
    for model in models_to_plot:
        print(f"\n{'='*80}")
        print(f"Processing model: {pc.MODEL_NAMES.get(model, model)}")
        print(f"{'='*80}")

        # Collect recall data
        print("Loading reasoning data and computing recall per generator...")
        recall_data = collect_recall_data_for_model(model)

        # Create plots
        print("Creating bar plots...")
        create_recall_bar_plots(model, recall_data)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
