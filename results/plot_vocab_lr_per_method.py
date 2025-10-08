"""
Vocabulary LR Per-Method Stacked Bar Charts

Generates publication-ready horizontal bar charts showing top 10 positive words
for each method (baseline, cot, s2) with percentage improvement in correctness odds.

Layout:
- One figure per VLM model (3 total)
- 3 horizontal subplots: baseline | cot | s2
- Top 10 positive words (highest OR at top)
- Bar length = (OR - 1) × 100% (improvement in correctness odds)
- Method-specific colors from plot_config.py

Output:
- figures/{model}_vocab_lr_per_method.pdf (3 files)

Usage:
    python results/plot_vocab_lr_per_method.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared plotting config
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "vocab_lr_per_model_method"

# Plot settings
TOP_N = 20  # Number of words to display per subplot

# Figure dimensions (2-column ACL format)
FIGURE_SIZE = (pc.ACL_FULL_WIDTH, 3.5)  # (width, height) in inches

# =============================================================================
# DATA LOADING
# =============================================================================

def load_method_results(model_name, method_name):
    """Load vocabulary LR results for a specific model-method combination.

    Args:
        model_name: Model identifier (e.g., 'qwen25-vl-7b')
        method_name: Method identifier (e.g., 'baseline', 'cot', 's2')

    Returns:
        dict: JSON data with top_positive_words, or None if file not found
    """
    json_path = RESULTS_DIR / f"{model_name}_{method_name}_vocab_lr.json"

    if not json_path.exists():
        print(f"WARNING: Results file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

def extract_top_words(data, top_n=10):
    """Extract top N positive words with odds ratios.

    Args:
        data: JSON data from vocab LR analysis
        top_n: Number of top words to extract

    Returns:
        tuple: (words, improvement_pcts) where improvement_pcts = (OR - 1) * 100
    """
    if data is None or 'top_positive_words' not in data:
        return [], []

    top_words = data['top_positive_words'][:top_n]

    words = [item['word'] for item in top_words]
    odds_ratios = [item['odds_ratio'] for item in top_words]

    # Calculate percentage improvement: (OR - 1) * 100
    improvement_pcts = [(or_val - 1) * 100 for or_val in odds_ratios]

    return words, improvement_pcts

# =============================================================================
# PLOTTING
# =============================================================================

def plot_model_methods(model_name):
    """Create 3-subplot figure for a single model showing all methods.

    Args:
        model_name: Model identifier (e.g., 'qwen25-vl-7b')
    """
    # Set publication style
    pc.set_publication_style()

    # Create figure with 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=False)

    # Iterate over methods
    for ax, method in zip(axes, pc.METHOD_ORDER):
        # Load data
        data = load_method_results(model_name, method)
        words, improvement_pcts = extract_top_words(data, top_n=TOP_N)

        if not words:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   fontsize=pc.AXIS_LABEL_FONT_SIZE, transform=ax.transAxes)
            ax.set_title(pc.METHOD_NAMES[method], fontsize=pc.TITLE_FONT_SIZE, fontweight='bold')
            continue

        # Reverse order so highest OR is at top
        words = words[::-1]
        improvement_pcts = improvement_pcts[::-1]

        # Y positions for bars
        y_positions = np.arange(len(words))

        # Plot horizontal bars
        ax.barh(y_positions, improvement_pcts,
               height=0.7,  # Increased to reduce gaps between bars
               color=pc.METHOD_COLORS[method],
               edgecolor='black',
               linewidth=pc.BAR_EDGE_WIDTH)

        # Set y-axis with word labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(words, fontsize=pc.TICK_LABEL_FONT_SIZE)
        ax.set_ylim(-0.5, len(words) - 0.5)  # Tighten y-axis limits

        # Set x-axis label only for middle subplot
        if method == 'cot':
            ax.set_xlabel('Increase in Odds of Correctness (\\%)',
                         fontsize=pc.AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis='x', labelsize=pc.TICK_LABEL_FONT_SIZE)

        # Grid
        ax.grid(axis='x', alpha=pc.GRID_ALPHA, linestyle=pc.GRID_LINESTYLE,
               linewidth=pc.GRID_LINEWIDTH, color=pc.GRID_COLOR)
        ax.set_axisbelow(True)

        # Subplot title
        ax.set_title(pc.METHOD_NAMES[method], fontsize=pc.TITLE_FONT_SIZE, fontweight='bold')

        # Set x-axis to start at 0
        ax.set_xlim(left=0)

        # Set spine (box) styling
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = pc.FIGURES_DIR / f"{model_name}_vocab_lr_per_method.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=pc.PUBLICATION_DPI)
    print(f"✓ Saved: {output_path}")

    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate vocab LR per-method plots for all models."""
    print("=" * 80)
    print("VOCABULARY LR PER-METHOD PLOTTING")
    print("=" * 80)
    print(f"Input directory: {RESULTS_DIR}")
    print(f"Output directory: {pc.FIGURES_DIR}")
    print(f"Models: {pc.MODEL_ORDER}")
    print(f"Methods: {pc.METHOD_ORDER}")
    print("=" * 80)

    # Check if input directory exists
    if not RESULTS_DIR.exists():
        print(f"ERROR: Input directory not found: {RESULTS_DIR}")
        print("Please run calculate_vocab_lr.py with PER_MODEL_METHOD_ANALYSIS=True first")
        sys.exit(1)

    # Generate plot for each model
    for model_name in pc.MODEL_ORDER:
        print(f"\nGenerating plot for {pc.MODEL_NAMES[model_name]}...")
        plot_model_methods(model_name)

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
