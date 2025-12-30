#!/usr/bin/env python3
"""
Aggregate Confidence Metrics from Performance JSON Files

Loads avg_token_prob_stats and macro_f1 metrics across 5 intervals
(0.00, 0.25, 0.50, 0.75, 1.00) from performance_{timestamp}_with_probs.json
files and aggregates them into a summary table.

Usage:
    python aggregate_margins.py
"""

import json
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ['genimage-2k', 'df40-2k', 'd3-2k']
MODELS = ['qwen25-vl-7b', 'llava-onevision-7b', 'llama32-vision-11b']
PHRASES = ['baseline', 'cot', 's2']
MODES = ['prefill', 'prompt']

# =============================================================================
# DATA LOADING
# =============================================================================

def find_latest_performance_probs_file(dataset: str, model: str, phrase: str, mode: str) -> Optional[Path]:
    """Find the most recent performance_{timestamp}_with_probs.json file for given configuration."""
    # Baseline ignores mode
    if phrase == 'baseline':
        output_dir = config.get_output_dir(dataset, model, phrase, mode='prefill', n=1)
    else:
        output_dir = config.get_output_dir(dataset, model, phrase, mode, n=1)

    probs_files = sorted(output_dir.glob("performance_*_with_probs.json"))

    if not probs_files:
        return None

    return probs_files[-1]


def load_confidence_statistics(dataset: str, model: str, phrase: str, mode: str) -> Optional[dict]:
    """
    Load confidence statistics from performance JSON.

    Returns:
        Dict with interval metrics or None if file not found
    """
    perf_file = find_latest_performance_probs_file(dataset, model, phrase, mode)

    if perf_file is None:
        return None

    try:
        with open(perf_file, 'r') as f:
            data = json.load(f)

        intervals = data.get('intervals', {})
        if not intervals:
            return None

        # Extract avg_token_prob_stats['mean'] and macro_f1 for each interval
        result = {}
        for interval_label in ['0.00', '0.25', '0.50', '0.75', '1.00']:
            if interval_label in intervals:
                interval_data = intervals[interval_label]
                avg_prob_stats = interval_data.get('avg_token_prob_stats', {})
                result[f'avg_prob_{interval_label}'] = avg_prob_stats.get('mean')
                result[f'macro_f1_{interval_label}'] = interval_data.get('macro_f1')
                result[f'count_{interval_label}'] = avg_prob_stats.get('count', 0)
            else:
                result[f'avg_prob_{interval_label}'] = None
                result[f'macro_f1_{interval_label}'] = None
                result[f'count_{interval_label}'] = 0

        return result
    except Exception as e:
        print(f"⚠️  Error loading {perf_file}: {e}")
        return None


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_all_configurations():
    """Aggregate interval statistics across all configurations."""
    results = []

    for dataset in DATASETS:
        for model in MODELS:
            for phrase in PHRASES:
                # Baseline only has one mode
                if phrase == 'baseline':
                    modes_to_check = ['prefill']
                else:
                    modes_to_check = MODES

                for mode in modes_to_check:
                    interval_stats = load_confidence_statistics(dataset, model, phrase, mode)

                    if interval_stats is not None:
                        row = {
                            'dataset': dataset,
                            'model': model,
                            'phrase': phrase,
                            'mode': mode,
                            'status': 'OK'
                        }
                        # Add all interval metrics
                        row.update(interval_stats)
                        results.append(row)

                        # Print summary (using final interval 1.00)
                        avg_prob_final = interval_stats.get('avg_prob_1.00')
                        macro_f1_final = interval_stats.get('macro_f1_1.00')
                        count = interval_stats.get('count_1.00', 0)
                        avg_prob_str = f"{avg_prob_final:.4f}" if avg_prob_final is not None else "N/A"
                        macro_f1_str = f"{macro_f1_final:.4f}" if macro_f1_final is not None else "N/A"
                        print(f"✓ {dataset:15s} / {model:20s} / {phrase:10s} / {mode:25s} | "
                              f"Final AvgProb: {avg_prob_str}, Final F1: {macro_f1_str}, Count: {count}")
                    else:
                        row = {
                            'dataset': dataset,
                            'model': model,
                            'phrase': phrase,
                            'mode': mode,
                            'status': 'MISSING'
                        }
                        # Add None for all interval metrics
                        for interval_label in ['0.00', '0.25', '0.50', '0.75', '1.00']:
                            row[f'avg_prob_{interval_label}'] = None
                            row[f'macro_f1_{interval_label}'] = None
                            row[f'count_{interval_label}'] = 0
                        results.append(row)
                        print(f"✗ {dataset:15s} / {model:20s} / {phrase:10s} / {mode:25s} | MISSING")

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame):
    """Save results to CSV and text file."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = results_dir / "aggregate_intervals.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n💾 Saved CSV: {csv_path}")

    # Save formatted text table
    txt_path = results_dir / "aggregate_intervals.txt"
    with open(txt_path, 'w') as f:
        f.write("="*150 + "\n")
        f.write("AGGREGATE INTERVAL METRICS (Avg Token Probability & Macro F1 across 5 intervals)\n")
        f.write("="*150 + "\n\n")

        # Summary statistics
        f.write("SUMMARY:\n")
        f.write(f"  Total configurations: {len(df)}\n")
        f.write(f"  Found: {(df['status'] == 'OK').sum()}\n")
        f.write(f"  Missing: {(df['status'] == 'MISSING').sum()}\n")
        f.write(f"  Intervals: 0.00 (0%), 0.25 (25%), 0.50 (50%), 0.75 (75%), 1.00 (100%)\n")
        f.write("\n")

        # Group by dataset
        for dataset in DATASETS:
            df_dataset = df[df['dataset'] == dataset]
            if df_dataset.empty:
                continue

            f.write(f"\nDataset: {dataset}\n")
            f.write("-"*150 + "\n")

            # Create table for this dataset
            for model in MODELS:
                df_model = df_dataset[df_dataset['model'] == model]
                if df_model.empty:
                    continue

                f.write(f"\n  Model: {model}\n")
                for _, row in df_model.iterrows():
                    if row['status'] == 'OK':
                        mode_str = f"({row['mode']})" if row['phrase'] != 'baseline' else ""
                        f.write(f"    {row['phrase']:15s} {mode_str:25s}\n")

                        # Avg Token Probability progression
                        f.write(f"      AvgProb: ")
                        for interval in ['0.00', '0.25', '0.50', '0.75', '1.00']:
                            prob = row[f'avg_prob_{interval}']
                            if prob is not None:
                                f.write(f"{interval}={prob:.4f}  ")
                            else:
                                f.write(f"{interval}=N/A  ")
                        f.write(f"\n")

                        # Macro F1 progression
                        f.write(f"      MacroF1: ")
                        for interval in ['0.00', '0.25', '0.50', '0.75', '1.00']:
                            f1 = row[f'macro_f1_{interval}']
                            if f1 is not None:
                                f.write(f"{interval}={f1:.4f}  ")
                            else:
                                f.write(f"{interval}=N/A  ")
                        f.write(f"  Count: {row['count_1.00']}\n")
                        f.write("\n")
                    else:
                        mode_str = f"({row['mode']})" if row['phrase'] != 'baseline' else ""
                        f.write(f"    {row['phrase']:15s} {mode_str:25s} | MISSING\n")

        # List missing configurations
        missing_df = df[df['status'] == 'MISSING']
        if not missing_df.empty:
            f.write("\n" + "="*150 + "\n")
            f.write("MISSING CONFIGURATIONS:\n")
            f.write("="*150 + "\n")
            for _, row in missing_df.iterrows():
                f.write(f"  • {row['dataset']} / {row['model']} / {row['phrase']} / {row['mode']}\n")

        f.write("\n" + "="*150 + "\n")

    print(f"💾 Saved formatted table: {txt_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main aggregation pipeline."""
    print("="*120)
    print("INTERVAL METRICS AGGREGATION (Avg Token Probability & Macro F1)")
    print("="*120)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Phrases: {PHRASES}")
    print(f"Modes: {MODES} (baseline uses only prefill)")
    print(f"Intervals: 0.00 (0%), 0.25 (25%), 0.50 (50%), 0.75 (75%), 1.00 (100%)")
    print("="*120)
    print()

    # Aggregate all configurations
    df = aggregate_all_configurations()

    if df.empty:
        print("\n❌ No data found!")
        return

    # Display summary
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print(f"Total configurations: {len(df)}")
    print(f"Found: {(df['status'] == 'OK').sum()}")
    print(f"Missing: {(df['status'] == 'MISSING').sum()}")

    # Save results
    save_results(df)

    print("\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()
