"""
Vocabulary Logistic Regression Analysis for VLM Reasoning

Trains logistic regression models to identify vocabulary that predicts correct answers
across prompting strategies (baseline, cot, s2). Supports both global (all models combined)
and per-model analysis.

Features:
- Train/validation split for hyperparameter tuning
- Model-specific hierarchical caching
- Consistent vocabulary across train/val/combined
- spaCy lemmatization with sklearn stopwords
- Parallel preprocessing with nlp.pipe

Output:
- Global mode: results/vocab_lr_global/vocab_lr_global.json
- Per-model mode: results/vocab_lr_per_model/{model}_vocab_lr.json

Usage:
    # Global analysis (all models combined)
    python results/calculate_vocab_lr.py

    # Per-model analysis (set PER_MODEL_ANALYSIS = True in config)
"""

import sys
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import spacy
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score

# =============================================================================
# SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# Configuration
PHRASES_TO_ANALYZE = ['baseline', 'cot', 's2']
ALL_MODELS = ['qwen25-vl-7b', 'llava-onevision-7b', 'llama32-vision-11b']
TRAIN_DATASETS = ['df40', 'd3', 'genimage']
VAL_DATASETS = ['df40-2k', 'd3-2k', 'genimage-2k']
MODE = 'prefill'
N = 1

# Analysis mode
PER_MODEL_ANALYSIS = True  # True: per-model LR, False: global LR
PER_MODEL_METHOD_ANALYSIS = True  # True: per-model-method LR, overrides PER_MODEL_ANALYSIS

# Hyperparameters
N_WORKERS = 16
TOP_N_WORDS = 20
MAX_FEATURES = 10000
MIN_DF = 500
PENALTY = 'l1'
SOLVER = 'liblinear'
PARAM_GRID = {
    'C': [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005,
          0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
}

# Paths
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR_GLOBAL = RESULTS_DIR / "vocab_lr_global"
OUTPUT_DIR_PER_MODEL = RESULTS_DIR / "vocab_lr_per_model"
CACHE_FILE = RESULTS_DIR / "preprocessed_cache.pkl"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / "calculate_vocab_lr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# spaCy setup
logger.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
logger.info("spaCy model loaded successfully")

# =============================================================================
# SKIP WORDS
# =============================================================================

def extract_skip_words_from_text(text: str) -> Set[str]:
    """Extract lemmatized tokens from text for skip word set."""
    if not text:
        return set()
    skip_words = set()
    try:
        doc = nlp(text.lower())
        for token in doc:
            if token.is_alpha and len(token.text) > 1:
                skip_words.add(token.lemma_)
    except Exception as e:
        logger.warning(f"Error extracting skip words: {e}")
    return skip_words

def build_skip_word_set() -> Set[str]:
    """Build skip word set from sklearn stopwords, question, and phrase words."""
    skip_words = set(ENGLISH_STOP_WORDS)
    skip_words.update(extract_skip_words_from_text(config.BASE_QUESTION))
    for phrase_name in PHRASES_TO_ANALYZE:
        phrase_text = config.get_phrase_text(phrase_name)
        if phrase_text:
            skip_words.update(extract_skip_words_from_text(phrase_text))
    logger.info(f"Total skip words: {len(skip_words)}")
    return skip_words

SKIP_WORDS = build_skip_word_set()

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_responses_parallel(response_texts: List[str]) -> List[str]:
    """Preprocess responses using spaCy's optimized batch processing."""
    if not response_texts:
        return []

    # Lowercase all texts first
    lowercased_texts = [text.lower() if isinstance(text, str) else "" for text in response_texts]
    cleaned_texts = []

    # Use nlp.pipe for efficient batch processing with multiprocessing
    with tqdm(total=len(lowercased_texts), desc="Preprocessing", unit="response") as pbar:
        for doc in nlp.pipe(lowercased_texts, batch_size=1000, n_process=N_WORKERS):
            tokens = [token.lemma_ for token in doc
                     if token.is_alpha and len(token.text) > 1
                     and token.lemma_ not in SKIP_WORDS]
            cleaned_texts.append(' '.join(tokens))
            pbar.update(1)

    return cleaned_texts

# =============================================================================
# DATA LOADING
# =============================================================================

def find_latest_reasoning_file(model: str, phrase: str, dataset: str) -> Path:
    """Find latest reasoning JSON for model/phrase/dataset."""
    output_dir = config.get_output_dir(dataset, model, phrase, mode=MODE, n=N)
    reasoning_files = sorted(output_dir.glob("reasoning_*.json"), reverse=True)
    if not reasoning_files:
        raise FileNotFoundError(f"No reasoning files for {model}/{phrase}/{dataset} in {output_dir}")
    return reasoning_files[0]

def load_data_for_model(model: str, datasets: List[str], method_filter: str = None) -> Tuple[List[str], List[int], List[str]]:
    """Load reasoning texts, labels, and method labels for a specific model from given datasets.

    Args:
        model: Model name
        datasets: List of dataset names to load
        method_filter: Optional method name to filter by (e.g., 'baseline', 'cot', 's2')
    """
    all_texts, all_labels, all_methods = [], [], []

    for phrase in PHRASES_TO_ANALYZE:
        if method_filter and phrase != method_filter:
            continue
        for dataset in datasets:
            try:
                json_path = find_latest_reasoning_file(model, phrase, dataset)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                for example in data:
                    if 'responses' in example and example['responses']:
                        response = example['responses'][0]
                        reasoning_text = response.get('reasoning_response', '')
                        score = response.get('score', 0)
                        if reasoning_text:
                            all_texts.append(reasoning_text)
                            all_labels.append(score)
                            all_methods.append(phrase)
            except FileNotFoundError as e:
                logger.error(f"FATAL: {e}")
                sys.exit(1)

    logger.info(f"{model}: {len(all_texts)} responses | {Counter(all_labels)} | {Counter(all_methods)}")
    return all_texts, all_labels, all_methods

# =============================================================================
# CACHING
# =============================================================================

def get_cache_key_str(model_name: str, datasets: List[str]) -> str:
    """Generate unique cache key string for model + datasets combination."""
    datasets_str = '-'.join(sorted(datasets))
    return f"{model_name}_{datasets_str}"

def get_cache_validation_key(model_name: str, datasets: List[str]) -> Dict[str, Any]:
    """Generate cache validation key for config checking."""
    return {
        'model': model_name,
        'skip_words': len(SKIP_WORDS),
        'phrases': PHRASES_TO_ANALYZE,
        'datasets': sorted(datasets)
    }

def try_load_cache(model_name: str, datasets: List[str]) -> Tuple[bool, List[str], List[int], List[str], List[str]]:
    """Try to load cache for specific model+datasets. Returns (success, cleaned_texts, labels, method_labels, raw_texts)."""
    if not CACHE_FILE.exists():
        return False, [], [], [], []

    try:
        with open(CACHE_FILE, 'rb') as f:
            all_cache = pickle.load(f)

        cache_key = get_cache_key_str(model_name, datasets)
        if cache_key not in all_cache:
            return False, [], [], [], []

        entry = all_cache[cache_key]
        expected_validation = get_cache_validation_key(model_name, datasets)

        if entry.get('cache_key') != expected_validation:
            logger.info(f"Cache invalid for {cache_key}: config changed")
            return False, [], [], [], []

        logger.info(f"✓ Cache hit for {cache_key}: {len(entry['cleaned_texts']):,} responses")
        return True, entry['cleaned_texts'], entry['labels'], entry['method_labels'], entry['raw_texts']
    except Exception as e:
        logger.warning(f"Cache load failed for {model_name}: {e}")
        return False, [], [], [], []

def save_to_cache(model_name: str, datasets: List[str], cleaned_texts, labels, method_labels, raw_texts):
    """Save preprocessed data to cache for specific model+datasets, preserving other entries."""
    try:
        all_cache = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'rb') as f:
                all_cache = pickle.load(f)

        cache_key = get_cache_key_str(model_name, datasets)
        all_cache[cache_key] = {
            'cache_key': get_cache_validation_key(model_name, datasets),
            'cleaned_texts': cleaned_texts,
            'labels': labels,
            'method_labels': method_labels,
            'raw_texts': raw_texts,
            'timestamp': datetime.now().isoformat()
        }

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(all_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✓ Saved cache for {cache_key}: {len(cleaned_texts):,} responses")
    except Exception as e:
        logger.warning(f"Cache save failed for {model_name}: {e}")

# =============================================================================
# MODEL TRAINING
# =============================================================================

def create_logistic_regression(C: float, n_jobs: int = 1) -> LogisticRegression:
    """Create LogisticRegression model with standard hyperparameters."""
    return LogisticRegression(
        penalty=PENALTY,
        solver=SOLVER,
        C=C,
        max_iter=100000,
        random_state=0,
        n_jobs=n_jobs,
        class_weight='balanced'
    )

def tune_hyperparameters(X_train, y_train, X_val, y_val) -> Tuple[float, Dict[str, Any]]:
    """Tune C parameter by training on train set, selecting best based on val set."""
    from joblib import Parallel, delayed

    logger.info(f"Tuning C parameter on validation set ({len(PARAM_GRID['C'])} values, {N_WORKERS} workers)...")

    def train_and_evaluate(C_value):
        """Train model with given C and return validation macro F1."""
        model = create_logistic_regression(C_value, n_jobs=1)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        return C_value, val_macro_f1

    # Parallel search over C values
    results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(train_and_evaluate)(C) for C in PARAM_GRID['C']
    )

    # Find best C
    best_C, best_val_macro_f1 = max(results, key=lambda x: x[1])
    logger.info(f"Best C: {best_C} with validation macro F1: {best_val_macro_f1:.4f}")

    # Train final model with best C and get detailed metrics
    best_model = create_logistic_regression(best_C, n_jobs=N_WORKERS)
    best_model.fit(X_train, y_train)

    # Detailed validation metrics
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
    val_macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_cm = confusion_matrix(y_val, y_val_pred)

    tuning_results = {
        'best_C': best_C,
        'val_macro_f1': best_val_macro_f1,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_confusion_matrix': {
            'TP': int(val_cm[1, 1]), 'TN': int(val_cm[0, 0]),
            'FP': int(val_cm[0, 1]), 'FN': int(val_cm[1, 0])
        },
        'all_C_results': results
    }

    logger.info(f"Best C: {best_C} | Val Macro F1: {best_val_macro_f1:.4f} | Val Acc: {val_accuracy:.4f}")
    return best_C, tuning_results

def train_final_model(X, y, best_C: float) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train final model on combined data with best hyperparameters."""
    logger.info(f"Training final model with C={best_C}...")

    model = create_logistic_regression(best_C, n_jobs=N_WORKERS)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    cm = confusion_matrix(y, y_pred)

    performance = {
        'C': best_C,
        'train_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'TP': int(cm[1, 1]), 'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]), 'FN': int(cm[1, 0])
        }
    }

    logger.info(f"Final model: Train Acc={accuracy:.4f}")
    return model, performance

# =============================================================================
# WORD EXTRACTION
# =============================================================================

def calculate_word_frequencies(X_combined, method_labels: List[str],
                               vocabulary: List[str]) -> Dict[str, Dict[str, int]]:
    """Calculate raw counts of responses containing each word, per method.

    Args:
        X_combined: Sparse binary matrix [n_documents, n_features]
        method_labels: List of method names for each document
        vocabulary: List of words corresponding to matrix columns
    """
    # Convert to dense for easier indexing (still efficient for small vocab subsets)
    X_dense = X_combined.toarray()

    # Create method masks
    method_masks = {}
    for method in PHRASES_TO_ANALYZE:
        method_masks[method] = np.array([m == method for m in method_labels])

    # Calculate frequencies per word per method
    word_freqs = {}
    for word_idx, word in enumerate(vocabulary):
        word_freqs[word] = {}
        for method in PHRASES_TO_ANALYZE:
            if method not in method_masks:
                word_freqs[word][method] = 0
                continue
            # Count documents where word appears (column word_idx) AND method matches
            count = int(X_dense[method_masks[method], word_idx].sum())
            word_freqs[word][method] = count

    return word_freqs

def extract_top_words(model, vocabulary, X_combined, method_labels) -> Dict[str, Any]:
    """Extract top positive/negative words with frequencies.

    Args:
        model: Trained LogisticRegression model
        vocabulary: List of words (features)
        X_combined: Sparse binary matrix [n_documents, n_features]
        method_labels: List of method names for each document
    """
    coefficients = model.coef_[0]
    word_coef_pairs = list(zip(vocabulary, coefficients))

    positive_pairs = sorted([(w, c) for w, c in word_coef_pairs if c > 0],
                           key=lambda x: x[1], reverse=True)[:TOP_N_WORDS]
    negative_pairs = sorted([(w, c) for w, c in word_coef_pairs if c < 0],
                           key=lambda x: x[1])[:TOP_N_WORDS]

    logger.info(f"Top {TOP_N_WORDS} POSITIVE words (predict correct):")
    for i, (word, coef) in enumerate(positive_pairs[:20], 1):
        odds_ratio = np.exp(coef)
        logger.info(f"  {i:2d}. {word:20s} coef={coef:+.4f} OR={odds_ratio:.4f}")

    logger.info(f"\nTop {TOP_N_WORDS} NEGATIVE words (predict incorrect):")
    for i, (word, coef) in enumerate(negative_pairs[:20], 1):
        odds_ratio = np.exp(coef)
        logger.info(f"  {i:2d}. {word:20s} coef={coef:+.4f} OR={odds_ratio:.4f}")

    all_top_vocab = [w for w, _ in positive_pairs] + [w for w, _ in negative_pairs]
    word_freqs = calculate_word_frequencies(X_combined, method_labels, all_top_vocab)

    return {
        'top_positive_words': [
            {
                'rank': i + 1,
                'word': word,
                'odds_ratio': round(float(np.exp(coef)), 4),
                'count_s2': word_freqs[word]['s2'],
                'count_cot': word_freqs[word]['cot'],
                'count_baseline': word_freqs[word]['baseline']
            }
            for i, (word, coef) in enumerate(positive_pairs)
        ],
        'top_negative_words': [
            {
                'rank': i + 1,
                'word': word,
                'odds_ratio': round(float(np.exp(coef)), 4),
                'count_s2': word_freqs[word]['s2'],
                'count_cot': word_freqs[word]['cot'],
                'count_baseline': word_freqs[word]['baseline']
            }
            for i, (word, coef) in enumerate(negative_pairs)
        ]
    }

# =============================================================================
# CORE LR WORKFLOW
# =============================================================================

def run_lr_analysis(train_texts: List[str], train_labels: List[int], train_methods: List[str],
                    val_texts: List[str], val_labels: List[int], val_methods: List[str],
                    model_name: str = None, vectorizer: CountVectorizer = None) -> Dict[str, Any]:
    """
    Core LR workflow: vectorize, tune, train, extract words.
    Works for both global (model_name=None) and per-model analysis.

    Args:
        vectorizer: Optional pre-fitted vectorizer for shared vocabulary across methods.
    """
    prefix = f"[{model_name}] " if model_name else ""

    # Combine train + val
    combined_texts = train_texts + val_texts
    combined_labels = train_labels + val_labels
    combined_methods = train_methods + val_methods

    # Vectorization (fit new or use shared)
    if vectorizer is None:
        logger.info(f"{prefix}Building vocabulary from combined data...")
        vectorizer = CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, binary=True)
        vectorizer.fit(combined_texts)
    else:
        logger.info(f"{prefix}Using shared vocabulary...")

    vocabulary = vectorizer.get_feature_names_out()

    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_combined = vectorizer.transform(combined_texts)
    logger.info(f"{prefix}Vocab: {len(vocabulary):,} | Train: {X_train.shape} | Val: {X_val.shape}")

    # Hyperparameter tuning
    logger.info(f"{prefix}Tuning hyperparameters...")
    best_C, tuning_results = tune_hyperparameters(X_train, np.array(train_labels),
                                                  X_val, np.array(val_labels))

    # Final model
    logger.info(f"{prefix}Training final model...")
    final_model, final_performance = train_final_model(X_combined, np.array(combined_labels), best_C)

    # Extract top words
    logger.info(f"{prefix}Extracting top words...")
    top_words = extract_top_words(final_model, vocabulary, X_combined, combined_methods)

    # Build results
    results = {
        **top_words,
        'tuning_results': tuning_results,
        'final_model_performance': final_performance,
        'hyperparameters': {
            'penalty': PENALTY,
            'solver': SOLVER,
            'max_features': MAX_FEATURES,
            'min_df': MIN_DF,
            'top_n_words': TOP_N_WORDS
        },
        'dataset_info': {
            'train_responses': len(train_texts),
            'val_responses': len(val_texts),
            'combined_responses': len(combined_texts),
            'train_datasets': TRAIN_DATASETS,
            'val_datasets': VAL_DATASETS,
            'correct': int(sum(combined_labels)),
            'incorrect': int(len(combined_labels) - sum(combined_labels)),
            'method_distribution': dict(Counter(combined_methods))
        }
    }

    if model_name:
        results['model'] = model_name

    return results

# =============================================================================
# MAIN
# =============================================================================

def load_model_data(model_name: str, datasets: List[str], method_filter: str = None) -> Tuple[List[str], List[int], List[str]]:
    """Load and preprocess data for a model, with caching.

    Args:
        model_name: Model name
        datasets: List of dataset names to load
        method_filter: Optional method name to filter by (e.g., 'baseline', 'cot', 's2')
    """
    cache_hit, cleaned_texts, labels, method_labels, raw_texts = try_load_cache(model_name, datasets)

    if not cache_hit:
        # Always load ALL methods for caching (never filter during cache build)
        raw_texts, labels, method_labels = load_data_for_model(model_name, datasets, method_filter=None)
        logger.info(f"Preprocessing {model_name}...")
        cleaned_texts = preprocess_responses_parallel(raw_texts)
        save_to_cache(model_name, datasets, cleaned_texts, labels, method_labels, raw_texts)

    # Apply method filter after cache load if needed
    if method_filter:
        filtered = [(t, l, m) for t, l, m in zip(cleaned_texts, labels, method_labels) if m == method_filter]
        if filtered:
            cleaned_texts, labels, method_labels = zip(*filtered)
        else:
            cleaned_texts, labels, method_labels = [], [], []

    return list(cleaned_texts), list(labels), list(method_labels)

def run_global_analysis():
    """Run global LR analysis across all models."""
    logger.info("=" * 80)
    logger.info("GLOBAL LOGISTIC REGRESSION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Models: {ALL_MODELS}")
    logger.info(f"Train datasets: {TRAIN_DATASETS}")
    logger.info(f"Val datasets: {VAL_DATASETS}")
    logger.info("=" * 80)

    # Load train data from all models
    logger.info("\n[TRAIN DATA] Loading from all models...")
    train_texts_all, train_labels_all, train_methods_all = [], [], []
    for model_name in ALL_MODELS:
        texts, labels, methods = load_model_data(model_name, TRAIN_DATASETS)
        train_texts_all.extend(texts)
        train_labels_all.extend(labels)
        train_methods_all.extend(methods)
    logger.info(f"✓ Train: {len(train_texts_all):,} responses")

    # Load val data from all models
    logger.info("\n[VALIDATION DATA] Loading from all models...")
    val_texts_all, val_labels_all, val_methods_all = [], [], []
    for model_name in ALL_MODELS:
        texts, labels, methods = load_model_data(model_name, VAL_DATASETS)
        val_texts_all.extend(texts)
        val_labels_all.extend(labels)
        val_methods_all.extend(methods)
    logger.info(f"✓ Val: {len(val_texts_all):,} responses")

    # Run LR analysis
    results = run_lr_analysis(train_texts_all, train_labels_all, train_methods_all,
                             val_texts_all, val_labels_all, val_methods_all)

    # Save results
    OUTPUT_DIR_GLOBAL.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR_GLOBAL / "vocab_lr_global.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("GLOBAL ANALYSIS COMPLETED")
    logger.info(f"✓ Results: {output_file}")
    logger.info(f"  Val Macro F1: {results['tuning_results']['val_macro_f1']:.4f}")
    logger.info(f"  Val Acc: {results['tuning_results']['val_accuracy']:.4f}")
    logger.info(f"  Final Train Acc: {results['final_model_performance']['train_accuracy']:.4f}")
    logger.info("=" * 80)

def run_per_model_analysis():
    """Run separate LR analysis for each model."""
    logger.info("=" * 80)
    logger.info("PER-MODEL LOGISTIC REGRESSION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Models: {ALL_MODELS}")
    logger.info(f"Train datasets: {TRAIN_DATASETS}")
    logger.info(f"Val datasets: {VAL_DATASETS}")
    logger.info("=" * 80)

    OUTPUT_DIR_PER_MODEL.mkdir(parents=True, exist_ok=True)

    for model_name in ALL_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING: {model_name}")
        logger.info(f"{'='*80}")

        # Load data
        train_texts, train_labels, train_methods = load_model_data(model_name, TRAIN_DATASETS)
        val_texts, val_labels, val_methods = load_model_data(model_name, VAL_DATASETS)

        # Run LR analysis
        results = run_lr_analysis(train_texts, train_labels, train_methods,
                                 val_texts, val_labels, val_methods,
                                 model_name=model_name)

        # Save results
        output_file = OUTPUT_DIR_PER_MODEL / f"{model_name}_vocab_lr.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ {model_name} results: {output_file}")
        logger.info(f"  Val Macro F1: {results['tuning_results']['val_macro_f1']:.4f}")
        logger.info(f"  Val Acc: {results['tuning_results']['val_accuracy']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PER-MODEL ANALYSIS COMPLETED")
    logger.info("=" * 80)

def run_per_model_method_analysis():
    """Run separate LR analysis for each model-method combination with shared vocabulary per model."""
    logger.info("=" * 80)
    logger.info("PER-MODEL-METHOD LOGISTIC REGRESSION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Models: {ALL_MODELS}")
    logger.info(f"Methods: {PHRASES_TO_ANALYZE}")
    logger.info(f"Train datasets: {TRAIN_DATASETS}")
    logger.info(f"Val datasets: {VAL_DATASETS}")
    logger.info("=" * 80)

    OUTPUT_DIR_PER_MODEL_METHOD = RESULTS_DIR / "vocab_lr_per_model_method"
    OUTPUT_DIR_PER_MODEL_METHOD.mkdir(parents=True, exist_ok=True)

    for model_name in ALL_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*80}")

        # Load ALL methods for this model to fit shared vectorizer
        logger.info(f"Loading all methods for {model_name} to build shared vocabulary...")
        all_train_texts, _, _ = load_model_data(model_name, TRAIN_DATASETS)
        all_val_texts, _, _ = load_model_data(model_name, VAL_DATASETS)

        all_combined_texts = all_train_texts + all_val_texts

        # Fit shared vectorizer on ALL methods combined
        logger.info(f"Fitting shared vectorizer for {model_name} on all methods...")
        shared_vectorizer = CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, binary=True)
        shared_vectorizer.fit(all_combined_texts)
        vocabulary = shared_vectorizer.get_feature_names_out()
        logger.info(f"✓ Shared vocabulary size: {len(vocabulary):,} words")

        # Now train separate LR for each method using shared vectorizer
        for method in PHRASES_TO_ANALYZE:
            logger.info(f"\n{'='*80}")
            logger.info(f"ANALYZING: {model_name} - {method}")
            logger.info(f"{'='*80}")

            # Load data filtered by method
            train_texts, train_labels, train_methods = load_model_data(model_name, TRAIN_DATASETS, method_filter=method)
            val_texts, val_labels, val_methods = load_model_data(model_name, VAL_DATASETS, method_filter=method)

            logger.info(f"Train: {len(train_texts)} responses | Val: {len(val_texts)} responses")

            # Run LR analysis with shared vectorizer
            results = run_lr_analysis(train_texts, train_labels, train_methods,
                                     val_texts, val_labels, val_methods,
                                     model_name=f"{model_name}_{method}",
                                     vectorizer=shared_vectorizer)

            # Save results
            output_file = OUTPUT_DIR_PER_MODEL_METHOD / f"{model_name}_{method}_vocab_lr.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"✓ {model_name} - {method} results: {output_file}")
            logger.info(f"  Val Macro F1: {results['tuning_results']['val_macro_f1']:.4f}")
            logger.info(f"  Val Acc: {results['tuning_results']['val_accuracy']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PER-MODEL-METHOD ANALYSIS COMPLETED")
    logger.info("=" * 80)

def main():
    """Main entry point."""
    if PER_MODEL_METHOD_ANALYSIS:
        run_per_model_method_analysis()
    elif PER_MODEL_ANALYSIS:
        run_per_model_analysis()
    else:
        run_global_analysis()

if __name__ == "__main__":
    main()
