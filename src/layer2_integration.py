# src/layer2_integration.py

"""
Layer 2 Integration Module

This module provides the bridge between Layer 1's cached output and Layer 2's
offline analysis engine. It handles:
1. Loading Layer 1 cache files
2. Preparing data for Layer 2 consumption
3. Running complete Layer 2 grid search experiments
4. Aggregating results across queries and configurations
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

from src.utils import load_json, save_json
from src.layer2_analysis import (
    Layer2Config, Layer2Orchestrator, PTUMathEngine,
    run_layer2_experiments
)


logger = logging.getLogger(__name__)


def load_layer1_cache_combined(
    cache_dir: str,
    experiment_name: str = "default_experiment",
    top_k: int = 3,
    n_candidates: int = None
) -> Optional[Dict[str, Any]]:
    """
    Load the combined Layer 1 cache for a given experiment configuration.
    
    Args:
        cache_dir: Directory where Layer 1 caches are stored
        experiment_name: Name of the experiment
        top_k: Number of retrieved samples
        n_candidates: Number of candidates (if None, will try to infer)
    
    Returns:
        Combined cache dictionary with all queries, or None if not found
    """
    from src.layer1_base_execution import _get_cache_filename, _get_cache_path, _cache_exists
    
    # Generate cache filename
    filename = _get_cache_filename(top_k, n_candidates or 0, experiment_name)
    cache_path = _get_cache_path(cache_dir, filename)
    
    if not _cache_exists(cache_path):
        logger.warning(f"Layer 1 cache not found: {cache_path}")
        return None
    
    try:
        cache_data = load_json(cache_path)
        logger.info(
            f"Loaded Layer 1 cache from: {cache_path}\n"
            f"  Metadata: {cache_data.get('metadata', {})}\n"
            f"  Queries: {len(cache_data.get('queries', {}))} cached"
        )
        return cache_data
    except Exception as e:
        logger.error(f"Failed to load Layer 1 cache: {e}")
        return None


def extract_layer1_states_for_layer2(
    combined_cache: Dict[str, Any],
    query_indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Extract individual Layer 1 states from combined cache for Layer 2 processing.
    
    Args:
        combined_cache: The combined cache from Layer 1
        query_indices: Optional list of specific query indices to extract.
                      If None, extracts all queries.
    
    Returns:
        List of Layer 1 states ready for Layer 2 consumption
    """
    states = []
    queries_dict = combined_cache.get('queries', {})
    
    if query_indices is None:
        # Extract all
        query_indices = [int(k) for k in queries_dict.keys()]
    
    for idx in query_indices:
        key = str(idx)
        if key in queries_dict:
            state = queries_dict[key]
            state['target_query_idx'] = idx
            states.append(state)
            logger.debug(f"Extracted Layer 1 state for query #{idx}")
        else:
            logger.warning(f"Query #{idx} not found in cache")
    
    logger.info(f"Extracted {len(states)} Layer 1 states for Layer 2 processing")
    return states


def create_layer2_config_from_dict(config_dict: Dict[str, Any]) -> Layer2Config:
    """
    Create a Layer2Config instance from a dictionary.
    
    Args:
        config_dict: Dictionary with Layer 2 configuration keys
    
    Returns:
        Layer2Config instance
    """
    config = Layer2Config()
    
    # Block toggles
    if 'run_block_A' in config_dict:
        config.run_block_A = config_dict['run_block_A']
    if 'run_block_B' in config_dict:
        config.run_block_B = config_dict['run_block_B']
    if 'run_block_C' in config_dict:
        config.run_block_C = config_dict['run_block_C']
    
    # Global settings
    if 'evaluator_masking' in config_dict:
        config.evaluator_masking = config_dict['evaluator_masking']
    if 'base_scoring_strategies' in config_dict:
        config.base_scoring_strategies = config_dict['base_scoring_strategies']
    if 'global_pass_at_N' in config_dict:
        config.global_pass_at_N = config_dict['global_pass_at_N']
    if 'activation_threshold' in config_dict:
        config.activation_threshold = config_dict['activation_threshold']
    
    # Block A settings
    if 'block_A_strategies' in config_dict:
        config.block_A_strategies = config_dict['block_A_strategies']
    if 'block_A_weight_taker' in config_dict:
        config.block_A_weight_taker = config_dict['block_A_weight_taker']
    if 'block_A_weight_maker' in config_dict:
        config.block_A_weight_maker = config_dict['block_A_weight_maker']
    if 'top_ks_group' in config_dict:
        config.top_ks_group = config_dict['top_ks_group']
    
    # Block B settings
    if 'dynamic_k_methods' in config_dict:
        config.dynamic_k_methods = config_dict['dynamic_k_methods']
    if 'block_B_weight_taker' in config_dict:
        config.block_B_weight_taker = config_dict['block_B_weight_taker']
    if 'block_B_weight_maker' in config_dict:
        config.block_B_weight_maker = config_dict['block_B_weight_maker']
    if 'run_boundary_intersection_test' in config_dict:
        config.run_boundary_intersection_test = config_dict['run_boundary_intersection_test']
    
    # Block C settings
    if 'coverage_perspectives' in config_dict:
        config.coverage_perspectives = config_dict['coverage_perspectives']
    if 'block_C_tiebreaker_weight_taker' in config_dict:
        config.block_C_tiebreaker_weight_taker = config_dict['block_C_tiebreaker_weight_taker']
    if 'block_C_tiebreaker_weight_maker' in config_dict:
        config.block_C_tiebreaker_weight_maker = config_dict['block_C_tiebreaker_weight_maker']
    
    return config


def run_layer2_complete_pipeline(
    layer1_cache_dir: str,
    layer2_output_dir: str,
    experiment_name: str,
    layer2_config_dict: Optional[Dict[str, Any]] = None,
    query_indices: Optional[List[int]] = None,
    top_k: int = 3,
    n_candidates: int = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Complete end-to-end Layer 2 execution:
    Load Layer 1 cache -> Extract states -> Run Layer 2 grid search -> Generate report
    
    Args:
        layer1_cache_dir: Directory containing Layer 1 cache files
        layer2_output_dir: Directory to save Layer 2 results
        experiment_name: Name of the experiment
        layer2_config_dict: Dictionary with Layer 2 configuration (or None for defaults)
        query_indices: Specific query indices to process (or None for all)
        top_k: Number of retrieved samples (for cache filename)
        n_candidates: Number of candidates (for cache filename)
    
    Returns:
        Tuple of (all_results, master_report)
    """
    os.makedirs(layer2_output_dir, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("LAYER 2 EXECUTION PIPELINE")
    logger.info(f"{'='*80}")
    
    # Step 1: Load Layer 1 cache
    logger.info(f"\nStep 1: Loading Layer 1 cache from {layer1_cache_dir}")
    combined_cache = load_layer1_cache_combined(
        layer1_cache_dir, experiment_name, top_k, n_candidates
    )
    
    if combined_cache is None:
        logger.error("Failed to load Layer 1 cache. Aborting Layer 2 execution.")
        return None, None
    
    # Step 2: Extract Layer 1 states
    logger.info(f"\nStep 2: Extracting Layer 1 states for Layer 2")
    layer1_states = extract_layer1_states_for_layer2(combined_cache, query_indices)
    
    if not layer1_states:
        logger.error("No Layer 1 states extracted. Aborting Layer 2 execution.")
        return None, None
    
    # Step 3: Prepare Layer 2 configuration
    logger.info(f"\nStep 3: Preparing Layer 2 configuration")
    if layer2_config_dict is None:
        layer2_config_dict = {}
    
    layer2_config = create_layer2_config_from_dict(layer2_config_dict)
    logger.info(f"Layer 2 Config: {layer2_config}")
    
    # Step 4: Run Layer 2 experiments
    logger.info(f"\nStep 4: Running Layer 2 experiments")
    all_results, master_report = run_layer2_experiments(
        layer1_states, layer2_config, layer2_output_dir
    )
    
    # Step 5: Save report and results
    logger.info(f"\nStep 5: Saving results")
    results_filepath = os.path.join(layer2_output_dir, "layer2_experiments_detailed.json")
    report_filepath = os.path.join(layer2_output_dir, "layer2_master_report.json")
    
    # Serialize results
    results_data = [
        {
            'target_query_idx': r.target_query_idx,
            'target_query_text': r.target_query_text,
            'ground_truth_answer': r.ground_truth_answer,
            'evaluator_setting': r.evaluator_setting,
            'scoring_strategy': r.scoring_strategy,
            'weight_taker': r.weight_taker,
            'weight_maker': r.weight_maker,
            'application': r.application,
            'subset_size': r.subset_size,
            'selected_candidates': r.selected_candidates,
            'list_ap_score': r.list_ap_score,
            'group_pass_at_n': r.group_pass_at_n,
            'timestamp': r.timestamp,
            'notes': r.notes
        }
        for r in all_results
    ]
    
    save_json(results_data, results_filepath)
    save_json(master_report, report_filepath)
    
    logger.info(f"\n{'='*80}")
    logger.info("LAYER 2 EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to:")
    logger.info(f"  - Detailed Results: {results_filepath}")
    logger.info(f"  - Master Report: {report_filepath}")
    logger.info(f"  - Total Experiments: {len(all_results)}")
    logger.info(f"  - Total Queries: {len(set(r.target_query_idx for r in all_results))}")
    
    return all_results, master_report


def validate_layer1_cache_structure(cache_data: Dict[str, Any]) -> bool:
    """
    Validate that Layer 1 cache has all required fields for Layer 2 processing.
    
    Args:
        cache_data: The combined cache dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_top_level = {'metadata', 'queries'}
    if not all(key in cache_data for key in required_top_level):
        logger.error(f"Missing top-level keys: {required_top_level}")
        return False
    
    # Check at least one query exists
    queries = cache_data.get('queries', {})
    if not queries:
        logger.error("No queries found in cache")
        return False
    
    # Spot-check first query
    first_query_key = list(queries.keys())[0]
    first_query = queries[first_query_key]
    
    required_query_fields = {
        'retrieved_set', 'candidate_set', 'intrinsic_baselines',
        'cross_evaluation_matrix', 'ground_truth_labels'
    }
    
    if not all(key in first_query for key in required_query_fields):
        logger.error(f"Query missing required fields: {required_query_fields}")
        return False
    
    logger.info("Layer 1 cache structure validated successfully")
    return True


def print_layer1_cache_summary(cache_data: Dict[str, Any]) -> None:
    """Print a summary of the Layer 1 cache contents."""
    metadata = cache_data.get('metadata', {})
    queries = cache_data.get('queries', {})
    
    print(f"\n{'='*70}")
    print("LAYER 1 CACHE SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment: {metadata.get('experiment_name', 'N/A')}")
    print(f"Top-K: {metadata.get('top_k', 'N/A')}")
    print(f"Number of Candidates: {metadata.get('n_candidates', 'N/A')}")
    print(f"Total Queries Cached: {len(queries)}")
    print(f"Generated: {metadata.get('generation_timestamp', 'N/A')}")
    
    if queries:
        print(f"\nQuery Indices: {sorted([int(k) for k in queries.keys()])}")
        
        # Sample first query structure
        first_key = list(queries.keys())[0]
        first_query = queries[first_key]
        print(f"\nSample Query Structure (Query #{first_key}):")
        print(f"  - Retrieved Set Size: {len(first_query.get('retrieved_set', []))}")
        print(f"  - Candidate Set Size: {len(first_query.get('candidate_set', []))}")
        print(f"  - Cross-Eval Matrix Entries: {len(first_query.get('cross_evaluation_matrix', {}))}")
        print(f"  - Ground-Truth Labels: {len(first_query.get('ground_truth_labels', {}))}")
    
    print(f"{'='*70}\n")
