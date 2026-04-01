#!/usr/bin/env python3
"""
Analyze Stage 1 reasoning response lengths (token counts) per phrase × mode.

Loads reasoning JSONs, batch-tokenizes with each model's own tokenizer,
and saves per-config stats to JSON.

Usage:
    python results/analyze_response_length.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import helpers

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
MODELS = ['qwen25-vl-7b', 'llava-onevision-7b', 'qwen3-vl-8b']
PHRASES = ['baseline', 'cot', 's2']
MODES = ['prefill', 'prompt', 'prefill-pseudo-system']
N = 1

MODEL_DISPLAY = {
    'qwen25-vl-7b': 'Qwen2.5',
    'llava-onevision-7b': 'LLaVA',
    'qwen3-vl-8b': 'Qwen3',
}
DATASET_DISPLAY = {
    'd3-2k': 'D3', 'df40-2k': 'DF40', 'genimage-2k': 'GenImage',
}

# =============================================================================
# ANALYSIS
# =============================================================================

def _compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute length stats from an array of token counts."""
    if len(arr) == 0:
        return {'count': 0, 'mean_tokens': 0, 'median_tokens': 0, 'std_tokens': 0,
                'p5': 0, 'p25': 0, 'p75': 0, 'p95': 0}
    return {
        'count': len(arr),
        'mean_tokens': float(np.mean(arr)),
        'median_tokens': float(np.median(arr)),
        'std_tokens': float(np.std(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
    }


def main():
    print("=" * 90)
    print("RESPONSE LENGTH ANALYSIS (TOKEN COUNTS)")
    print("=" * 90)

    # Load tokenizers once
    tokenizers = {}
    for model in MODELS:
        print(f"  Loading tokenizer for {MODEL_DISPLAY[model]}...")
        model_config = config.get_model_config(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            model_config['hf_path'],
            trust_remote_code=model_config.get('trust_remote_code', False),
        )

    print()
    rows = []

    # Collect raw token counts for pooled per-mode stats (CoT + S2 combined)
    # Key: (dataset, model, mode) -> list of token counts
    pooled_tokens = {}

    for dataset in DATASETS:
        for model in MODELS:
            for phrase in PHRASES:
                modes = ['prefill'] if phrase == 'baseline' else MODES
                for mode in modes:
                    label = f"{phrase}/{mode}" if phrase != 'baseline' else 'baseline'

                    try:
                        data = helpers.load_reasoning_json(dataset, model, phrase, mode, N)
                    except FileNotFoundError:
                        print(f"  MISSING: {DATASET_DISPLAY[dataset]:>8s} / {MODEL_DISPLAY[model]:>5s} / {label}")
                        continue

                    # Tokenize raw texts to get individual counts
                    texts = []
                    for example in data:
                        for resp in example['responses']:
                            reasoning = resp['reasoning_response']
                            if reasoning:
                                texts.append(reasoning)

                    tokenizer = tokenizers[model]
                    if texts:
                        encoded = tokenizer(texts, add_special_tokens=False)
                        token_counts = [len(ids) for ids in encoded['input_ids']]
                    else:
                        token_counts = []

                    arr = np.array(token_counts) if token_counts else np.array([])
                    stats = _compute_stats(arr)

                    method = 'Baseline' if phrase == 'baseline' else {'cot': 'CoT', 's2': 'S2'}[phrase]
                    mode_label = '-' if phrase == 'baseline' else mode

                    rows.append({
                        'dataset': dataset,
                        'model': model,
                        'method': method,
                        'mode': mode_label,
                        'n_responses': stats['count'],
                        'mean_tokens': stats['mean_tokens'],
                        'median_tokens': stats['median_tokens'],
                        'std_tokens': stats['std_tokens'],
                        'p5': stats['p5'],
                        'p25': stats['p25'],
                        'p75': stats['p75'],
                        'p95': stats['p95'],
                    })

                    print(f"  OK: {DATASET_DISPLAY[dataset]:>8s} / {MODEL_DISPLAY[model]:>5s} / {label:>12s}  "
                          f"n={stats['count']:>5d}  mean={stats['mean_tokens']:.0f}  "
                          f"median={stats['median_tokens']:.0f}  std={stats['std_tokens']:.0f}")

                    # Accumulate for pooled mode stats (skip baseline)
                    if phrase != 'baseline':
                        pool_key = (dataset, model, mode)
                        pooled_tokens.setdefault(pool_key, []).extend(token_counts)

    # Compute pooled per-mode rows (CoT + S2 combined)
    print("\n  --- Pooled per-mode (CoT + S2 combined) ---")
    for (dataset, model, mode), counts in sorted(pooled_tokens.items()):
        arr = np.array(counts) if counts else np.array([])
        stats = _compute_stats(arr)

        rows.append({
            'dataset': dataset,
            'model': model,
            'method': 'Pooled',
            'mode': mode,
            'n_responses': stats['count'],
            'mean_tokens': stats['mean_tokens'],
            'median_tokens': stats['median_tokens'],
            'std_tokens': stats['std_tokens'],
            'p5': stats['p5'],
            'p25': stats['p25'],
            'p75': stats['p75'],
            'p95': stats['p95'],
        })

        print(f"  OK: {DATASET_DISPLAY[dataset]:>8s} / {MODEL_DISPLAY[model]:>5s} / {'Pooled/'+mode:>22s}  "
              f"n={stats['count']:>5d}  mean={stats['mean_tokens']:.0f}  "
              f"median={stats['median_tokens']:.0f}  std={stats['std_tokens']:.0f}")

    # Save
    out_path = Path(__file__).resolve().parent / "response_length.json"
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
