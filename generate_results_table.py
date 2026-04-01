#!/usr/bin/env python3
"""
Generate Markdown results tables from performance JSON files.

Reads performance JSONs for all model/dataset/method combinations and
prints two tables: one for Macro F1 and one for Accuracy.
"""

import json
import glob
from pathlib import Path

BASE_PATH = Path("/data3/zkachwal/zeroshot-pgt/output")

MODELS = ["qwen3-vl-8b", "qwen25-vl-7b", "llava-onevision-7b"]
MODEL_DISPLAY = {
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "qwen25-vl-7b": "Qwen2.5-VL-7B",
    "llava-onevision-7b": "LLaVA-OneVision-7B",
}

DATASETS = ["d3", "df40", "genimage"]
DATASET_DISPLAY = {"d3": "D3", "df40": "DF40", "genimage": "GenImage"}

# (phrase, mode, display_name)
METHODS = [
    ("baseline", None, "Baseline"),
    ("cot", "prefill", "CoT-prefill"),
    ("cot", "prompt", "CoT-prompt"),
    ("cot", "prefill-pseudo-system", "CoT-pseudo-sys"),
    ("s2", "prefill", "S2-prefill"),
    ("s2", "prompt", "S2-prompt"),
    ("s2", "prefill-pseudo-system", "S2-pseudo-sys"),
]


def find_latest_performance_json(directory):
    """Find the latest performance JSON file in a directory.
    
    Looks for files matching performance_*.json or performance_fixed_*.json.
    Returns the one with the latest timestamp (lexicographic sort works
    because timestamps are YYYYMMDD_HHMMSS format).
    """
    patterns = [
        str(directory / "performance_fixed_*.json"),
        str(directory / "performance_*.json"),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # Sort lexicographically -- latest timestamp is last
    all_files.sort()
    return Path(all_files[-1])


def get_directory(dataset, model, phrase, mode):
    """Build the directory path for a given combination."""
    if phrase == "baseline":
        return BASE_PATH / dataset / model / "baseline"
    else:
        return BASE_PATH / dataset / model / phrase / mode


def extract_metrics(json_path):
    """Extract accuracy and macro_f1 from a performance JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        return {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        }
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"  Warning: Error reading {json_path}: {e}")
        return None


def collect_all_results():
    """Collect results for all combinations."""
    results = {}  # (model, method_name) -> {dataset: {accuracy, macro_f1}}
    missing = []

    for model in MODELS:
        for phrase, mode, method_name in METHODS:
            key = (model, method_name)
            results[key] = {}

            for dataset in DATASETS:
                directory = get_directory(dataset, model, phrase, mode)
                json_path = find_latest_performance_json(directory)

                if json_path is None:
                    missing.append(f"  {MODEL_DISPLAY[model]} / {DATASET_DISPLAY[dataset]} / {method_name}: no file in {directory}")
                    results[key][dataset] = None
                    continue

                metrics = extract_metrics(json_path)
                if metrics is None:
                    results[key][dataset] = None
                    continue

                results[key][dataset] = metrics

    return results, missing


def format_value(value):
    """Format a metric value as a percentage string or dash."""
    if value is None:
        return "-"
    return f"{value * 100:.1f}"


def find_best_per_column(results, metric_key):
    """Find the best value for each (model, dataset) combination."""
    best = {}
    for model in MODELS:
        for dataset in DATASETS:
            best_val = -1
            for _, _, method_name in METHODS:
                entry = results.get((model, method_name), {}).get(dataset)
                if entry is not None and entry.get(metric_key) is not None:
                    val = entry[metric_key]
                    if val > best_val:
                        best_val = val
            best[(model, dataset)] = best_val if best_val > -1 else None
    return best


def print_table(results, metric_key, metric_label):
    """Print a nicely formatted markdown table."""
    print(f"\n## {metric_label} (%) by Model, Method, and Dataset\n")

    # Header
    header = f"| {'Model':<22} | {'Method':<16} |"
    sep = f"|{'-'*24}|{'-'*18}|"
    for dataset in DATASETS:
        header += f" {DATASET_DISPLAY[dataset]:>8} |"
        sep += f"{'-'*10}|"
    print(header)
    print(sep)

    best = find_best_per_column(results, metric_key)

    for model in MODELS:
        for i, (phrase, mode, method_name) in enumerate(METHODS):
            key = (model, method_name)
            model_col = MODEL_DISPLAY[model] if i == 0 else ""

            row = f"| {model_col:<22} | {method_name:<16} |"
            for dataset in DATASETS:
                entry = results[key].get(dataset)
                if entry is not None and entry.get(metric_key) is not None:
                    val = entry[metric_key]
                    formatted = format_value(val)
                    # Bold the best value in each model/dataset group
                    if best.get((model, dataset)) is not None and abs(val - best[(model, dataset)]) < 1e-9:
                        formatted = f"**{formatted}**"
                    row += f" {formatted:>8} |"
                else:
                    row += f" {'-':>8} |"
            print(row)

        # Print separator between models
        print(sep)


def main():
    print("=" * 70)
    print("Results Table Generator")
    print("=" * 70)

    results, missing = collect_all_results()

    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for m in missing:
            print(m)

    print_table(results, "macro_f1", "Macro F1")
    print_table(results, "accuracy", "Accuracy")

    # Also print a compact summary of best methods per model/dataset
    print("\n## Best Method per Model/Dataset (Macro F1)\n")
    header = f"| {'Model':<22} |"
    sep = f"|{'-'*24}|"
    for dataset in DATASETS:
        header += f" {DATASET_DISPLAY[dataset]:>18} |"
        sep += f"{'-'*20}|"
    print(header)
    print(sep)

    for model in MODELS:
        row = f"| {MODEL_DISPLAY[model]:<22} |"
        for dataset in DATASETS:
            best_method = "-"
            best_val = -1
            for _, _, method_name in METHODS:
                entry = results.get((model, method_name), {}).get(dataset)
                if entry is not None and entry.get("macro_f1") is not None:
                    if entry["macro_f1"] > best_val:
                        best_val = entry["macro_f1"]
                        best_method = method_name
            cell = f"{best_method} ({best_val*100:.1f})" if best_val > -1 else "-"
            row += f" {cell:>18} |"
        print(row)
    print(sep)


if __name__ == "__main__":
    main()
