"""
Stacked Bar Chart for Vocabulary LR Results

Generates publication-ready stacked bar charts with positive and negative words
in top/bottom subplots. Creates one plot per model.

Layout:
- Top subplot: Positive words (sorted high to low)
  - Bar length = Odds ratio - 1
  - Bar composition = proportion of s2/cot/baseline counts
- Bottom subplot: Negative words (sorted low to high)
  - Bar length = 1 - Odds ratio
  - Bar composition = proportion of s2/cot/baseline counts
- Shared legend
- Words on Y-axis

Usage:
    python results/plot_vocab_lr.py
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
RESULTS_DIR = PROJECT_ROOT / "results"

# Plot settings
TOP_N = 20  # Number of words to display per chart

FIGURE_SIZE = (pc.ACL_COLUMN_WIDTH, 6)  # (width, height) in inches - fits ACL 1-column

# =============================================================================
# DATA LOADING
# =============================================================================

def load_model_results(model_name):
    """Load vocabulary LR results for a specific model."""
    json_path = RESULTS_DIR / "vocab_lr_per_model" / f"{model_name}_vocab_lr.json"

    if not json_path.exists():
        print(f"WARNING: Results file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

# =============================================================================
# PLOTTING
# =============================================================================

def plot_stacked_subplot(ax, words_data, is_positive=True, show_xlabel=True):
    """
    Create stacked bar chart in a subplot.

    Args:
        ax: Matplotlib axis object
        words_data: List of word dictionaries from JSON
        is_positive: If True, normal y-axis; if False, inverted y-axis
        show_xlabel: If True, show x-axis label
    """
    # Extract top N words
    top_words_data = words_data[:TOP_N]

    # Always reverse so highest magnitude is at top
    top_words_data = list(reversed(top_words_data))

    words = [item['word'] for item in top_words_data]
    odds_ratios = np.array([item['odds_ratio'] for item in top_words_data])
    count_s2 = np.array([item['count_s2'] for item in top_words_data])
    count_cot = np.array([item['count_cot'] for item in top_words_data])
    count_baseline = np.array([item['count_baseline'] for item in top_words_data])

    # Calculate total counts and proportions
    total_counts = count_s2 + count_cot + count_baseline
    prop_baseline = count_baseline / total_counts
    prop_cot = count_cot / total_counts
    prop_s2 = count_s2 / total_counts

    # Convert to percentage: (OR - 1) × 100
    # OR=1.5 → +50%, OR=0.7 → -30%
    bar_lengths = (odds_ratios - 1) * 100

    y_positions = np.arange(len(words)) * pc.WORD_SPACING

    # Stacked bars: baseline + cot + s2
    # Each segment length = proportion × total bar length
    baseline_lengths = prop_baseline * bar_lengths
    cot_lengths = prop_cot * bar_lengths
    s2_lengths = prop_s2 * bar_lengths

    # Plot stacked bars with different order for positive vs negative
    if is_positive:
        # Positive bars: baseline (closest to 0), then cot, then s2
        ax.barh(y_positions, baseline_lengths, height=pc.BAR_HEIGHT,
                color=pc.BASELINE_COLOR, label='Baseline', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)

        ax.barh(y_positions, cot_lengths, height=pc.BAR_HEIGHT,
                left=baseline_lengths, color=pc.COT_COLOR, label='CoT', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)

        ax.barh(y_positions, s2_lengths, height=pc.BAR_HEIGHT,
                left=baseline_lengths + cot_lengths, color=pc.S2_COLOR, label='S2', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)
    else:
        # Negative bars: reverse order so s2 (closest to 0), then cot, then baseline
        ax.barh(y_positions, s2_lengths, height=pc.BAR_HEIGHT,
                color=pc.S2_COLOR, label='S2', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)

        ax.barh(y_positions, cot_lengths, height=pc.BAR_HEIGHT,
                left=s2_lengths, color=pc.COT_COLOR, label='CoT', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)

        ax.barh(y_positions, baseline_lengths, height=pc.BAR_HEIGHT,
                left=s2_lengths + cot_lengths, color=pc.BASELINE_COLOR, label='Baseline', alpha=pc.BAR_ALPHA,
                edgecolor='black', linewidth=pc.BAR_EDGE_WIDTH)

    # WORD LABELS
    ax.set_yticks(y_positions)
    ax.set_yticklabels(words, fontsize=pc.TICK_LABEL_FONT_SIZE)

    # X-AXIS
    if is_positive:
        # Positive: bars extend right (0 to max)
        max_bar = bar_lengths.max()
        ax.set_xlim(0, max_bar * 1.1)
        # Y-axis on left (default)
        ax.yaxis.tick_left()
    else:
        # Negative: bars extend left (min to 0), and INVERT Y-AXIS
        min_bar = bar_lengths.min()
        ax.set_xlim(min_bar * 1.1, 0)
        ax.invert_yaxis()  # Flip y-axis so words read bottom to top
        # Y-axis on right (where bars start)
        ax.yaxis.tick_right()

    if show_xlabel:
        ax.set_xlabel('Change in detection (\\%)', fontsize=pc.AXIS_LABEL_FONT_SIZE)

    # Set tick label sizes explicitly
    ax.tick_params(axis='x', labelsize=pc.TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=pc.TICK_LABEL_FONT_SIZE)

    # GRID
    ax.grid(axis='x', alpha=pc.GRID_ALPHA, linestyle=pc.GRID_LINESTYLE, linewidth=pc.GRID_LINEWIDTH)
    ax.set_axisbelow(True)

    # SPINES
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def create_positive_plot(data, model_name):
    """
    Create plot with positive words only.

    Args:
        data: Dictionary with 'top_positive_words'
        model_name: Name of the model for title
    """
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]))

    # Plot positive words
    plot_stacked_subplot(ax, data['top_positive_words'], is_positive=True, show_xlabel=True)

    # Legend at top
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=pc.LEGEND_FONT_SIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for legend
    return fig


def create_negative_plot(data, model_name):
    """
    Create plot with negative words only.

    Args:
        data: Dictionary with 'top_negative_words'
        model_name: Name of the model for title
    """
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]))

    # Plot negative words
    plot_stacked_subplot(ax, data['top_negative_words'], is_positive=False, show_xlabel=True)

    # Legend at top
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=pc.LEGEND_FONT_SIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for legend
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Set publication-quality style (LaTeX rendering)
    pc.set_publication_style()

    print("=" * 80)
    print("VOCABULARY LR STACKED BAR CHARTS")
    print("=" * 80)

    for model_name in pc.MODEL_ORDER:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")

        # Load data
        data = load_model_results(model_name)
        if data is None:
            print(f"Skipping {model_name} (results not found)")
            continue

        print(f"Dataset info:")
        print(f"  Train responses: {data['dataset_info']['train_responses']:,}")
        print(f"  Val responses: {data['dataset_info']['val_responses']:,}")
        print(f"  Combined responses: {data['dataset_info']['combined_responses']:,}")
        print(f"  Best C: {data['final_model_performance']['C']}")
        print(f"  Final accuracy: {data['final_model_performance']['train_accuracy']:.4f}")
        print(f"  Final F1: {data['final_model_performance']['f1']:.4f}")

        # Create positive plot
        print(f"\nGenerating positive plot (top {TOP_N} positive words)...")
        fig_pos = create_positive_plot(data, model_name)
        output_pos_pdf = pc.FIGURES_DIR / f"vocab_lr_{model_name}_positive.pdf"
        print(f"  Saving: {output_pos_pdf}")
        fig_pos.savefig(output_pos_pdf, dpi=pc.PUBLICATION_DPI, bbox_inches='tight')
        plt.close(fig_pos)

        # Create negative plot
        print(f"Generating negative plot (top {TOP_N} negative words)...")
        fig_neg = create_negative_plot(data, model_name)
        output_neg_pdf = pc.FIGURES_DIR / f"vocab_lr_{model_name}_negative.pdf"
        print(f"  Saving: {output_neg_pdf}")
        fig_neg.savefig(output_neg_pdf, dpi=pc.PUBLICATION_DPI, bbox_inches='tight')
        plt.close(fig_neg)

    print("\n" + "=" * 80)
    print("PLOT GENERATION COMPLETED")
    print("=" * 80)
    for model_name in pc.MODEL_ORDER:
        print(f"  {pc.FIGURES_DIR / f'vocab_lr_{model_name}_positive.pdf'}")
        print(f"  {pc.FIGURES_DIR / f'vocab_lr_{model_name}_negative.pdf'}")

if __name__ == "__main__":
    main()
