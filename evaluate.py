#!/usr/bin/env python3
"""
Zero-shot PGT Evaluation Script

Clean, functional evaluation script for testing VLM models on AI-generated
image detection using two-stage prompting with prefill strategies.

Features:
- Three VLM models: Qwen2.5-VL, LLaVA-OneVision, Llama-3.2-Vision
- Four prefill modes: direct, pseudo, prompt, instruct
- Two-stage evaluation: reasoning generation + clean answer extraction
- Comprehensive metrics with reasoning trace storage

Usage:
    python evaluate.py --model qwen25-vl-7b --dataset df40 --phrase cot
    python evaluate.py --model llama32-vision-11b --dataset genimage --phrase cot --mode prefill-pseudo
    python evaluate.py --model llava-onevision-7b --dataset d3 --phrase cot --mode prompt
    python evaluate.py --model qwen25-vl-7b --dataset df40 --phrase cot --mode instruct
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Fix MKL threading layer incompatibility with multiprocessing
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Set vLLM worker multiprocessing method to 'spawn' for CUDA compatibility
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Disable vLLM's default logging configuration to avoid conflicts
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import helpers

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Zero-shot PGT Evaluation Script")

    parser.add_argument('--model', '-m', type=str, required=True, choices=config.get_supported_models(), help='VLM model to evaluate')
    parser.add_argument('--dataset', '-d', type=str, required=True, choices=config.get_supported_datasets(), help='Dataset to evaluate on')
    parser.add_argument('--phrase', '-p', type=str, required=True, choices=config.get_supported_phrases(), help='Phrase to use (e.g., cot, artifacts)')
    parser.add_argument('--mode', type=str, default=None, choices=config.PHRASE_MODES, help='Mode: prefill (append after template), prefill-pseudo-system (system: start with X), prefill-pseudo-user (user: start with X), prompt (append to question), instruct (system: X). Ignored for baseline phrase.')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory (default: output/{model_name})')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device IDs for VLLM (e.g., "0,1")')
    parser.add_argument('--n', type=int, default=1, help='Number of responses to generate per input (default: 1)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited examples')
    parser.add_argument('--override', action='store_true', help='Override existing results if they exist')

    return parser.parse_args()

def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed arguments
        logger: Logger instance

    Raises:
        ValueError: If arguments are invalid
    """
    try:
        config.validate_config_combination(args.model, args.dataset, args.phrase)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    # Set default mode if not provided
    if args.mode is None:
        args.mode = 'prefill'

    # Warn if baseline is used with explicit --mode (mode is ignored for baseline)
    if args.phrase == 'baseline' and args.mode != 'prefill':
        logger.warning(f"⚠️  --mode is ignored for baseline phrase (all modes are equivalent). Output will be in baseline/ directory.")

    # Check if dataset files exist
    try:
        dataset_config = config.get_dataset_config(args.dataset)
        if not dataset_config['csv_path'].exists():
            logger.error(f"Dataset CSV not found: {dataset_config['csv_path']}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        sys.exit(1)

def setup_environment(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Set up environment for evaluation.

    Args:
        args: Parsed arguments
        logger: Logger instance
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logger.info(f"CUDA_VISIBLE_DEVICES set to '{args.cuda}'")


def main():
    """Main evaluation pipeline."""

    # Parse and validate arguments
    args = parse_arguments()

    # Create logger early
    logger, log_path = helpers.create_logger(
        args.dataset, args.model, args.phrase, args.mode,
        logger_name='evaluation',
        log_suffix='',
        include_vllm_loggers=True,
        n=args.n
    )

    # Now validate and set up environment (passing logger)
    validate_arguments(args, logger)
    setup_environment(args, logger)

    logger.info("🚀 Starting Zero-shot PGT Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Phrase: {args.phrase} (mode: {args.mode})")
    logger.info(f"Number of responses: {args.n}")
    logger.info(f"Debug mode: {args.debug}")

    try:
        # Check if results already exist
        output_dir = Path(args.output_dir) if args.output_dir else config.get_output_dir(args.dataset, args.model, args.phrase, args.mode, n=args.n)

        # Look for any existing performance files in the output directory
        existing_performance_files = list(output_dir.glob("performance_*.json"))

        if existing_performance_files and not args.override:
            logger.warning(f"⚠️  Results already exist in {output_dir}")
            logger.warning(f"   Found {len(existing_performance_files)} existing performance file(s)")
            logger.warning(f"   Use --override flag to overwrite existing results")
            logger.info("❌ Evaluation skipped")
            return
        elif existing_performance_files and args.override:
            logger.info(f"♻️  Override mode: Will overwrite existing results in {output_dir}")

        # 1. Load and prepare dataset
        logger.info("📂 Loading dataset...")
        examples = helpers.load_dataset(args.dataset, logger)

        # Debug mode: use only first 5 examples
        if args.debug:
            examples = examples[:5]
            logger.info(f"🐛 Debug mode: Using only {len(examples)} examples")

        # 2. Build prompts with phrase
        logger.info("🔨 Building prompts...")
        examples = helpers.build_prompts_for_examples(
            examples, args.phrase, args.model, logger, args.mode
        )

        # 3. Apply chat templates and handle phrases
        logger.info("🛠️ Applying chat templates...")
        examples = helpers.apply_chat_template_and_phrase(examples, args.model, logger)

        # 4. Perform two-stage evaluation
        logger.info("🧠 Starting two-stage evaluation...")
        results = helpers.evaluate_two_stage(examples, args.model, logger, n=args.n)

        # 5. Calculate metrics
        logger.info("📊 Calculating metrics...")
        metrics = helpers.calculate_metrics(results, logger)

        # 6. Save results
        logger.info("💾 Saving results...")
        # output_dir already computed earlier for existence check

        reasoning_path, performance_path = helpers.save_results(
            results, metrics, output_dir, args.model, args.dataset, args.phrase, logger, args.mode, n=args.n
        )

        # 7. Display final results
        logger.info("✅ Evaluation completed successfully!")
        logger.info("\n" + "="*70)
        logger.info("🎯 ZERO-SHOT PGT EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Model: {args.model}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Phrase: {args.phrase} (mode: {args.mode})")
        logger.info(f"Examples: {metrics['total_examples']}")
        logger.info(f"Correct: {metrics['correct_predictions']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        logger.info("")
        logger.info("Per-class metrics:")
        for class_name, class_metrics in metrics['class_metrics'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {class_metrics['precision']:.4f}")
            logger.info(f"    Recall: {class_metrics['recall']:.4f}")
            logger.info(f"    F1-Score: {class_metrics['f1']:.4f}")
        logger.info("")
        logger.info("Confusion Matrix:")
        cm = metrics['confusion_matrix']
        logger.info(f"  True Positives (AI→AI): {cm['TP']}")
        logger.info(f"  True Negatives (Real→Real): {cm['TN']}")
        logger.info(f"  False Positives (Real→AI): {cm['FP']}")
        logger.info(f"  False Negatives (AI→Real): {cm['FN']}")
        logger.info("") 
        logger.info(f"Reasoning traces: {reasoning_path}")
        logger.info(f"Performance metrics: {performance_path}")
        logger.info(f"Evaluation logs: {log_path}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main()