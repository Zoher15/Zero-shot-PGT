#!/usr/bin/env python3
"""
Add Class Probabilities to Reasoning JSON

Extends reasoning JSON files with class probabilities computed from token groups
at the answer position (after reasoning + answer phrase).

Token Group Approach:
1. Define REAL_TOKENS and AI_TOKENS groups (with case/space variations)
2. For each token string: tokenize, skip whitespace/pollution ('a'), take first valid token
3. Deduplicate token IDs within each group (e.g., 'AI-generated' and 'AI-GENERATED' may share tokens)
4. Extract logprobs once per example
5. Sum probabilities across all unique tokens in each group
6. Compute margin: abs(real_group_prob - ai_group_prob)

Outputs:
- reasoning_{timestamp}_with_probs.json: Per-example class probabilities
- performance_{timestamp}_with_probs.json: Aggregate margin statistics + token mappings

Usage:
    python add_class_probs_to_reasoning.py --model qwen25-vl-7b --dataset df40 --phrase cot
    python add_class_probs_to_reasoning.py --model llama32-vision-11b --dataset genimage --phrase cot --mode prompt

Note: Only works with n=1 (single response per example)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

# Fix MKL threading layer incompatibility
os.environ.update({
    'MKL_THREADING_LAYER': 'GNU',
    'VLLM_WORKER_MULTIPROC_METHOD': 'fork',
    'VLLM_CONFIGURE_LOGGING': '0'
})

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import helpers
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pysbd

# =============================================================================
# TOKEN GROUPS
# =============================================================================

REAL_TOKENS = ['real', 'REAL', 'Real', ' real', ' REAL', ' Real', 'real.', 'REAL.', 'Real.', ' real.', ' REAL.', ' Real.']
AI_TOKENS = [
    'ai-generated', 'AI-generated', 'AI-Generated', 'AI-GENERATED',' ai-generated', ' AI-generated', ' AI-Generated', ' AI-GENERATED', 
    'ai-generated.', 'AI-generated.', 'AI-Generated.', 'AI-GENERATED.', ' ai-generated.', ' AI-generated.', ' AI-Generated.', ' AI-GENERATED.',
    'ai.', 'AI.', ' ai.', ' AI.']

# =============================================================================
# CLI & DATA LOADING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Add class probabilities to reasoning JSON")
    parser.add_argument('--model', '-m', required=True, choices=config.get_supported_models(),
                       help='VLM model used for evaluation')
    parser.add_argument('--dataset', '-d', required=True, choices=config.get_supported_datasets(),
                       help='Dataset evaluated')
    parser.add_argument('--phrase', '-p', required=True, choices=config.get_supported_phrases(),
                       help='Phrase strategy used')
    parser.add_argument('--mode', default='prefill', choices=config.PHRASE_MODES,
                       help='Phrase mode (default: prefill)')
    parser.add_argument('--cuda', default='0', help='CUDA device IDs (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Process only first 5 examples')
    return parser.parse_args()


# =============================================================================
# SENTENCE SPLITTING & INTERVAL CALCULATION
# =============================================================================

def split_reasoning_into_intervals(reasoning_text: str) -> Tuple[List[str], List[int]]:
    """
    Split reasoning text into 5 quarterly interval texts (0%, 25%, 50%, 75%, 100%).

    Args:
        reasoning_text: Full reasoning response text
        logger: Logger instance

    Returns:
        Tuple of (interval_texts, interval_sentence_counts)
        - interval_texts: List of 5 strings, each containing sentences up to that interval
        - interval_sentence_counts: List of 5 integers, number of sentences at each interval
    """
    # Use pysbd for robust sentence splitting
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(reasoning_text)

    num_sentences = len(sentences)

    # Calculate interval indices: 0, round(n*0.25), round(n*0.5), round(n*0.75), n
    interval_indices = [
        0,  # 0%
        round(num_sentences * 0.25),  # 25%
        round(num_sentences * 0.5),   # 50%
        round(num_sentences * 0.75),  # 75%
        num_sentences  # 100%
    ]

    # Build interval texts by joining sentences up to each index
    interval_texts = []
    for idx in interval_indices:
        if idx == 0:
            interval_texts.append("")  # No reasoning yet
        else:
            interval_texts.append(" ".join(sentences[:idx]))

    return interval_texts, interval_indices


# =============================================================================
# VLLM INPUT PREPARATION
# =============================================================================

def prepare_vllm_inputs(prompts: List[str], images: List[str], model_name: str, logger) -> Tuple[List[Dict], List[int]]:
    """
    Prepare vLLM inputs from prompts and images (reuses pattern from helpers._generate_stage1_responses).

    Args:
        prompts: List of prompt strings
        images: List of image paths
        model_name: Model name
        logger: Logger instance

    Returns:
        Tuple of (vLLM input dicts, valid_indices)
    """
    model_config = config.get_model_config(model_name)
    is_llava = 'llava' in model_config['hf_path'].lower()

    inputs = []
    valid_indices = []

    for i, (prompt, image_path) in enumerate(zip(prompts, images)):
        try:
            image = Image.open(image_path).convert('RGB')

            # Skip invalid images for LLaVA only (e.g., 1x1 pixels)
            if is_llava and (image.size[0] < 10 or image.size[1] < 10):
                logger.warning(f"⚠️ Skipping invalid image at index {i}: {image_path} (size: {image.size})")
                continue

            input_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "multi_modal_uuids": {"image": f"uuid_{i}"}
            }
            inputs.append(input_item)
            valid_indices.append(i)

        except Exception as e:
            if is_llava:
                logger.warning(f"⚠️ Skipping image at index {i} due to error: {str(e)}")
                continue
            else:
                raise

    if not inputs:
        raise ValueError("No valid images to process")

    return inputs, valid_indices


# =============================================================================
# TOKEN MAPPING SETUP
# =============================================================================

def build_token_mappings(tokenizer, logger) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build deduplicated token mappings for real and AI token groups.

    For each string in REAL_TOKENS and AI_TOKENS:
    1. Tokenize the string
    2. Skip if first token is whitespace (empty after strip)
    3. Skip if first token is 'a' or 'A' (pollutes results)
    4. Keep first valid token ID
    5. Deduplicate token IDs within each group

    Args:
        tokenizer: Tokenizer instance
        logger: Logger instance

    Returns:
        Tuple of (real_token_data, ai_token_data)
        Each data dict contains:
        - 'token_ids': List of deduplicated token IDs (used for computation)
        - 'mapping': Dict {original_string: {'decoded': decoded_str, 'id': token_id}}
    """
    def get_first_valid_token_id(token_string: str) -> Tuple[int, str]:
        """Get first valid token ID from string, or None if invalid."""
        token_ids = tokenizer.encode(token_string, add_special_tokens=False)
        if not token_ids:
            return None, None

        for token_id in token_ids:
            token_str = tokenizer.decode([token_id])
            stripped_lower = token_str.strip().lower()

            # Skip whitespace tokens
            if stripped_lower == '':
                continue

            # Skip 'a' tokens (pollutes results)
            if stripped_lower == 'a':
                return None, 'a (rejected)'

            # Found valid token
            return token_id, token_str

        return None, None

    # Build real token data
    real_mapping = {}
    real_token_ids = []
    real_seen_ids = set()

    for token_string in REAL_TOKENS:
        token_id, token_str = get_first_valid_token_id(token_string)
        if token_id is not None:
            # Always add to mapping (for documentation)
            real_mapping[token_string] = {'decoded': token_str, 'id': token_id}

            # Only add to token_ids list if not seen (deduplication)
            if token_id not in real_seen_ids:
                real_token_ids.append(token_id)
                real_seen_ids.add(token_id)

    # Build AI token data
    ai_mapping = {}
    ai_token_ids = []
    ai_seen_ids = set()

    for token_string in AI_TOKENS:
        token_id, token_str = get_first_valid_token_id(token_string)
        if token_id is not None:
            # Always add to mapping (for documentation)
            ai_mapping[token_string] = {'decoded': token_str, 'id': token_id}

            # Only add to token_ids list if not seen (deduplication)
            if token_id not in ai_seen_ids:
                ai_token_ids.append(token_id)
                ai_seen_ids.add(token_id)

    real_data = {
        'token_ids': real_token_ids,
        'mapping': real_mapping
    }

    ai_data = {
        'token_ids': ai_token_ids,
        'mapping': ai_mapping
    }

    logger.info(f"✅ Built token mappings - Real: {len(real_token_ids)} unique token IDs, AI: {len(ai_token_ids)} unique token IDs")
    logger.info(f"   Real mapping: {list(real_mapping.keys())}")
    logger.info(f"   AI mapping: {list(ai_mapping.keys())}")

    return real_data, ai_data


# =============================================================================
# LOGPROB EXTRACTION
# =============================================================================

def extract_logprobs_for_prompts(prompts: List[str], images: List[str], model_name: str,
                                 llm_engine, token_ids_to_keep: List[int], logger) -> List[Dict]:
    """
    Extract logprobs for immediate next token after each prompt, filtered to relevant token IDs.

    Args:
        prompts: List of prompt strings
        images: List of image paths
        model_name: Model name
        llm_engine: Pre-initialized LLM engine
        token_ids_to_keep: List of token IDs to filter (from real + AI token groups)
        logger: Logger instance

    Returns:
        List of filtered logprob dicts (one per example), only containing token IDs we care about
        Returns None for skipped examples (LLaVA invalid images)
    """
    model_config = config.get_model_config(model_name)
    is_llava = 'llava' in model_config['hf_path'].lower()

    inputs, valid_indices = prepare_vllm_inputs(prompts, images, model_name, logger)

    # Extract single token with all vocab logprobs
    sampling_params = SamplingParams(temperature=1.0, max_tokens=1, logprobs=-1, n=1)
    outputs = llm_engine.generate(inputs, sampling_params=sampling_params)

    # Filter logprobs to only keep token IDs we care about
    token_ids_set = set(token_ids_to_keep)

    # Map logprobs back to original indices (for LLaVA compatibility)
    if is_llava and len(valid_indices) < len(prompts):
        logprobs_list = [None] * len(prompts)  # Create list with None for all
        for valid_idx, output in zip(valid_indices, outputs):
            if output.outputs and output.outputs[0].logprobs:
                full_logprobs = output.outputs[0].logprobs[0]
                # Filter to only keep relevant token IDs
                filtered_logprobs = {tid: full_logprobs[tid] for tid in token_ids_set if tid in full_logprobs}
                logprobs_list[valid_idx] = filtered_logprobs
        logger.info(f"Extracted filtered logprobs for {len([lp for lp in logprobs_list if lp is not None])} examples ({len(prompts) - len(valid_indices)} skipped)")
    else:
        logprobs_list = []
        for output in outputs:
            if output.outputs and output.outputs[0].logprobs:
                full_logprobs = output.outputs[0].logprobs[0]
                # Filter to only keep relevant token IDs
                filtered_logprobs = {tid: full_logprobs[tid] for tid in token_ids_set if tid in full_logprobs}
                logprobs_list.append(filtered_logprobs)
            else:
                logprobs_list.append(None)
        logger.info(f"Extracted filtered logprobs for {len(logprobs_list)} examples")

    return logprobs_list


def compute_group_prob(token_logprobs: Dict, token_data: Dict[str, Any]) -> float:
    """
    Compute sum of probabilities for all tokens in a group.

    Args:
        token_logprobs: Dict mapping token_id -> logprob object
        token_data: Dict with 'token_ids' list and 'mapping' dict

    Returns:
        Sum of probabilities for all tokens in the group
    """
    total_prob = 0.0

    for token_id in token_data['token_ids']:
        if token_id in token_logprobs:
            logprob_obj = token_logprobs[token_id]
            # Check if logprob object is not None (can happen with some models/tokens)
            if logprob_obj is not None:
                logprob = logprob_obj.logprob
                prob = math.exp(logprob) if logprob != -float('inf') else 0.0
                total_prob += prob

    return total_prob


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def initialize_llm_and_tokenizer(model_name: str, logger) -> Tuple[LLM, AutoTokenizer]:
    """Initialize LLM engine and tokenizer."""
    logger.info("🚀 Initializing LLM engine for logprob extraction...")

    model_config = config.get_model_config(model_name)

    # Handle vLLM v1 engine flag
    if not model_config.get('use_v1_engine', True):
        os.environ.update({'VLLM_USE_V1': '0', 'MKL_SERVICE_FORCE_INTEL': '1'})
        logger.info("⚙️  Disabling vLLM v1 engine (using legacy v0 for compatibility)")

    llm_kwargs = {
        'model': model_config['hf_path'],
        'tensor_parallel_size': model_config['tensor_parallel_size'],
        'gpu_memory_utilization': model_config['gpu_memory_utilization'],
        'trust_remote_code': model_config['trust_remote_code'],
        'max_model_len': model_config.get('max_model_len', 16384),
        'max_num_seqs': model_config.get('max_num_seqs', 256),
        'max_logprobs': -1,  # Allow all vocab logprobs
        'seed': 0
    }

    logger.info("Setting max_logprobs=-1 to allow all vocabulary logprobs")
    llm_engine = LLM(**llm_kwargs)
    logger.info(f"✅ LLM engine initialized for {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config['hf_path'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    logger.info("✅ Tokenizer loaded")

    return llm_engine, tokenizer


def add_class_probs_to_examples(reasoning_data: List[Dict], model_name: str, mode: str, logger) -> Tuple[List[Dict], Dict[str, Any], Dict[str, Any]]:
    """
    Add class probabilities to reasoning data at 5 quarterly intervals using token group mappings.

    For each example, splits reasoning into 5 intervals (0%, 25%, 50%, 75%, 100%) and extracts
    logprobs after: full_prompt + interval_reasoning + answer_phrase + " This image is"

    The answer_phrase is included based on the mode (prefill modes include it, others may not).

    Args:
        reasoning_data: List of reasoning results (from reasoning JSON)
        model_name: Model name
        mode: Phrase mode (prefill, prefill-pseudo-system, prefill-pseudo-user, prompt, instruct)
        logger: Logger instance

    Returns:
        Tuple of (extended_reasoning_data, real_token_data, ai_token_data)
    """
    llm_engine, tokenizer = initialize_llm_and_tokenizer(model_name, logger)

    # Build token mappings once (deduplicated)
    logger.info("🔧 Building token mappings...")
    real_token_data, ai_token_data = build_token_mappings(tokenizer, logger)

    # Split reasoning into 5 quarterly intervals for each example
    logger.info("✂️  Splitting reasoning responses into quarterly intervals (0%, 25%, 50%, 75%, 100%)...")
    interval_labels = [0.00, 0.25, 0.50, 0.75, 1.00]

    # Pre-compute interval texts for all examples
    logger.info("📝 Pre-computing interval texts for all examples...")
    interval_data_by_example = []  # List of dicts with interval info per example

    for ex_idx, ex in enumerate(reasoning_data):
        reasoning_text = ex['responses'][0]['reasoning_response']
        interval_texts, interval_sentence_counts = split_reasoning_into_intervals(reasoning_text)

        interval_data_by_example.append({
            'full_prompt': ex['full_prompt'],
            'image': ex['image'],
            'interval_texts': interval_texts,
            'interval_sentence_counts': interval_sentence_counts
        })

    num_examples = len(reasoning_data)

    # Combine token IDs from both groups
    all_token_ids = real_token_data['token_ids'] + ai_token_data['token_ids']

    # Initialize interval storage for each example
    for ex in reasoning_data:
        ex['class_probs_intervals'] = []

    # Process each interval separately (better batching - similar lengths)
    for interval_idx, interval_label in enumerate(interval_labels):
        logger.info(f"📊 Processing interval {interval_label:.2f} ({interval_idx+1}/5)...")

        # Build prompts for this interval across all examples
        interval_prompts = []
        interval_images = []

        for ex_data in interval_data_by_example:
            interval_text = ex_data['interval_texts'][interval_idx]

            if interval_text:
                prompt = ex_data['full_prompt'] + interval_text + f" {config.ANSWER_PHRASE} This image is"
            else:
                # For interval 0.0 (no reasoning yet), just use the prompt
                if mode == 'prefill':
                    prompt = ex_data['full_prompt'] + f". {config.ANSWER_PHRASE} This image is"
                else:
                    prompt = ex_data['full_prompt'] + f" {config.ANSWER_PHRASE} This image is"

            interval_prompts.append(prompt)
            interval_images.append(ex_data['image'])

        # Extract logprobs for this interval batch
        logger.info(f"   Extracting logprobs for {len(interval_prompts)} examples at interval {interval_label:.2f}...")
        logprobs_list = extract_logprobs_for_prompts(interval_prompts, interval_images, model_name, llm_engine, all_token_ids, logger)

        # Compute class probabilities for each example at this interval
        for ex_idx, logprobs in enumerate(logprobs_list):
            sentence_count = interval_data_by_example[ex_idx]['interval_sentence_counts'][interval_idx]

            if logprobs is None:
                logger.warning(f"⚠️ Skipping example {ex_idx}, interval {interval_label:.2f} due to missing logprobs")
                interval_probs = {
                    'interval': interval_label,
                    'num_sentences': sentence_count,
                    'real_prob': None,
                    'ai_prob': None,
                    'margin': None,
                    'max_class_prob': None
                }
            else:
                # Compute group probabilities
                real_prob = compute_group_prob(logprobs, real_token_data)
                ai_prob = compute_group_prob(logprobs, ai_token_data)

                # Calculate confidence metrics
                margin = abs(real_prob - ai_prob)
                max_class_prob = max(real_prob, ai_prob)

                interval_probs = {
                    'interval': interval_label,
                    'num_sentences': sentence_count,
                    'real_prob': real_prob,
                    'ai_prob': ai_prob,
                    'margin': margin,
                    'max_class_prob': max_class_prob
                }

            reasoning_data[ex_idx]['class_probs_intervals'].append(interval_probs)

    logger.info(f"✅ Added class probabilities at 5 intervals for {num_examples} examples")
    return reasoning_data, real_token_data, ai_token_data


def _compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dict with mean, median, std, min, max, count
    """
    import numpy as np

    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }

    values_array = np.array(values)

    return {
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'count': len(values)
    }


def calculate_confidence_statistics(reasoning_data: List[Dict], logger) -> Dict[str, Dict[str, Any]]:
    """
    Calculate aggregate statistics for confidence metrics across all examples at each interval.

    Args:
        reasoning_data: List of reasoning results with class_probs_intervals
        logger: Logger instance

    Returns:
        Dict mapping interval (e.g., "0.00", "0.25") to {"margin_class_probs": {...}, "max_class_probs": {...}}
    """
    interval_labels = ["0.00", "0.25", "0.50", "0.75", "1.00"]
    interval_statistics = {}

    for interval_idx, interval_label in enumerate(interval_labels):
        # Extract metrics for this interval (skip examples with None values)
        margins = []
        max_class_probs = []

        for ex in reasoning_data:
            if 'class_probs_intervals' in ex and len(ex['class_probs_intervals']) > interval_idx:
                interval_data = ex['class_probs_intervals'][interval_idx]
                if interval_data['margin'] is not None:
                    margins.append(interval_data['margin'])
                    max_class_probs.append(interval_data['max_class_prob'])

        if not margins:
            logger.warning(f"⚠️ No valid confidence metrics found for interval {interval_label}")

        # Compute statistics for both metrics
        margin_stats = _compute_statistics(margins)
        max_class_probs_stats = _compute_statistics(max_class_probs)

        interval_statistics[interval_label] = {
            'margin_class_probs': margin_stats,
            'max_class_probs': max_class_probs_stats
        }

        logger.info(f"📊 Interval {interval_label} - Margin: {margin_stats['mean']:.4f} ± {margin_stats['std']:.4f}, Max prob: {max_class_probs_stats['mean']:.4f} ± {max_class_probs_stats['std']:.4f}")

    return interval_statistics


def save_extended_files(reasoning_data: List[Dict],
                       interval_statistics: Dict[str, Dict[str, Any]],
                       real_token_data: Dict[str, Any], ai_token_data: Dict[str, Any],
                       dataset_name: str, model_name: str, phrase_name: str, mode: str,
                       logger) -> Tuple[Path, Path]:
    """
    Save extended reasoning JSON and performance JSON with interval-based confidence statistics and token mappings.

    Args:
        reasoning_data: Extended reasoning data with class_probs_intervals
        interval_statistics: Dict mapping interval labels to margin/max_class_prob stats
        real_token_data: Real token data {'token_ids': [...], 'mapping': {...}}
        ai_token_data: AI token data {'token_ids': [...], 'mapping': {...}}
        dataset_name: Dataset name
        model_name: Model name
        phrase_name: Phrase name
        mode: Phrase mode
        logger: Logger instance

    Returns:
        Tuple of (reasoning_path, performance_path)
    """
    from datetime import datetime

    output_dir = config.get_output_dir(dataset_name, model_name, phrase_name, mode, n=1)

    # Find original reasoning file to extract timestamp
    reasoning_files = sorted([f for f in output_dir.glob("reasoning_*.json")
                             if '_with_probs' not in f.name])
    if reasoning_files:
        original_file = reasoning_files[-1]
        timestamp = original_file.stem.split('_', 1)[1]  # Extract 'YYYYMMDD_HHMMSS'
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save extended reasoning JSON
    reasoning_path = output_dir / f"reasoning_{timestamp}_with_probs.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(reasoning_data, f, indent=2)
    logger.info(f"💾 Saved extended reasoning JSON: {reasoning_path}")

    # Save performance JSON with interval-based confidence statistics and token mappings
    performance_data = {
        'model': model_name,
        'dataset': dataset_name,
        'phrase': phrase_name,
        'mode': mode,
        'n_responses': 1,
        'token_mappings': {
            'real_tokens': real_token_data,
            'ai_tokens': ai_token_data
        },
        'intervals': interval_statistics
    }

    performance_path = output_dir / f"performance_{timestamp}_with_probs.json"
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2)
    logger.info(f"💾 Saved performance with interval-based stats and token mappings: {performance_path}")

    return reasoning_path, performance_path


def main():
    """Main pipeline for adding class probabilities to reasoning JSON."""
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    logger, log_path = helpers.create_logger(
        args.dataset, args.model, args.phrase, args.mode,
        logger_name='class_probs', log_suffix='_probs',
        include_vllm_loggers=True, n=1
    )

    logger.info("🚀 Starting Class Probability Extraction")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Phrase: {args.phrase} (mode: {args.mode})")
    logger.info(f"CUDA devices: {args.cuda}")
    logger.info("Note: Only works with n=1 (single response per example)")

    try:
        # Load reasoning JSON using helpers
        logger.info("📂 Loading reasoning JSON...")
        reasoning_data = helpers.load_reasoning_json(
            args.dataset, args.model, args.phrase, args.mode, n=1
        )

        # Validate reasoning data structure (filter out malformed examples)
        logger.info("🔍 Validating reasoning data structure...")
        original_count = len(reasoning_data)
        valid_reasoning_data = []
        for i, ex in enumerate(reasoning_data):
            if 'responses' not in ex or not ex['responses']:
                logger.warning(f"⚠️ Skipping example {i} with missing/empty responses (image: {ex.get('image', 'unknown')})")
                continue
            if not ex['responses'][0].get('reasoning_response'):
                logger.warning(f"⚠️ Skipping example {i} with empty reasoning_response (image: {ex.get('image', 'unknown')})")
                continue
            valid_reasoning_data.append(ex)

        reasoning_data = valid_reasoning_data
        skipped_count = original_count - len(reasoning_data)
        if skipped_count > 0:
            logger.info(f"⚠️ Skipped {skipped_count} malformed examples (kept {len(reasoning_data)}/{original_count})")
        else:
            logger.info(f"✅ All {len(reasoning_data)} examples have valid structure")

        if args.debug:
            reasoning_data = reasoning_data[:5]
            logger.info(f"🐛 Debug mode: Processing only {len(reasoning_data)} examples")

        # Extract and add class probabilities
        logger.info("🧠 Extracting class probabilities...")
        extended_data, real_token_data, ai_token_data = add_class_probs_to_examples(
            reasoning_data, args.model, args.mode, logger
        )

        # Calculate confidence statistics per interval
        logger.info("📊 Calculating confidence statistics for each interval...")
        interval_statistics = calculate_confidence_statistics(extended_data, logger)

        # Save extended files
        logger.info("💾 Saving extended files...")
        reasoning_path, performance_path = save_extended_files(
            extended_data, interval_statistics, real_token_data, ai_token_data,
            args.dataset, args.model, args.phrase, args.mode, logger
        )

        # Display summary
        logger.info("\n" + "="*70)
        logger.info("✅ CLASS PROBABILITY EXTRACTION COMPLETED (INTERVAL-BASED)")
        logger.info("="*70)
        logger.info(f"Extended reasoning: {reasoning_path}")
        logger.info(f"Performance (interval stats): {performance_path}")
        logger.info(f"Examples processed: {len(extended_data)}")
        logger.info(f"Intervals analyzed: 0.00, 0.25, 0.50, 0.75, 1.00")
        # Show stats for final interval (1.00)
        if '1.00' in interval_statistics:
            final_stats = interval_statistics['1.00']
            logger.info(f"Final interval (1.00) avg margin: {final_stats['margin_class_probs']['mean']:.4f}")
            logger.info(f"Final interval (1.00) avg max prob: {final_stats['max_class_probs']['mean']:.4f}")
        logger.info(f"Log file: {log_path}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"💥 Extraction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
