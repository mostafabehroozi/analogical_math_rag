# src/layer2_analysis.py

"""
Layer 2: Application & Experimental Phase (Offline Analysis)

This module implements the complete offline analysis engine for the Advanced Analogical
Mirroring system. Layer 2 operates exclusively on cached Layer 1 state and produces NO
API calls. It serves as an automated scientific laboratory for systematic experimentation
with different scoring models and ranking strategies.

Core Architecture:
1. Math Engine: Computes PTU matrices from Layer 1 cached data
2. Evaluator Masking: Applies Self/Others/All evaluation settings
3. Scoring Strategies: Implements ScoreTake, ScoreMake, and Holistic scoring
4. Experimental Blocks: A (Baseline), B (Dynamic-K), C (Coverage)
5. Evaluation Harnesses: List-based (AP) and Group-based (Pass@N)
6. Report Generation: Comprehensive result aggregation

Key Design Principles:
- OFFLINE_ONLY: Zero API calls post Layer 1
- DETERMINISTIC: Reproducible results across runs
- ISOLATED: Each block has independent configurations
- SYSTEMATIC: Grid search across all base conditions and applications
- RIGOROUS: Standardized evaluation harnesses ensure fair comparison
"""

import logging
import os
import json
import csv
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import time
from datetime import datetime
from collections import defaultdict
import concurrent.futures

from src.utils import save_json, load_json, convert_numpy_for_json
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.evaluation import evaluate_single_answer_with_llm
from src.hf_sync import periodic_sync_check
from tqdm import tqdm
from src.prompts import create_final_reasoning_prompt, create_final_reasoning_prompt_simple, EXEMPLAR_FORMAT
from config import CONFIG as GLOBAL_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES & CONFIGURATION
# ============================================================================

@dataclass
class ExperimentResult:
    """Standardized result record for a single experimental run."""
    target_query_idx: int
    target_query_text: str
    ground_truth_answer: str
    
    # Base Conditions
    utility_calibration: str  # <-- I REMOVED the default value here
    evaluator_setting: str  # 'Self', 'Others', 'All'
    scoring_strategy: str   # 'ScoreTake', 'ScoreMake', 'Holistic'
    application: str = ""  # 'Block_A_Reranking', 'Block_A_TopK', 'Block_B_Dynamic', etc.
    weight_taker: float = 1.0
    weight_maker: float = 1.0
    subset_size: int = 0
    selected_candidates: List[str] = None 
    selected_evaluators: List[int] = None  
    selected_scores: List[float] = None  # NEW: Store the scores of chosen samples
    selected_exemplar_ids: List[str] = None    # NEW: Store the Parent Source IDs
    selected_exemplar_texts: List[str] = None  # NEW: Store the Parent Source Texts
    
    # Evaluation Metrics (Legacy)
    list_ap_score: Optional[float] = None  
    group_pass_at_n: Optional[float] = None  
    
# --- ACTIVE INFERENCE PAYLOADS (NEW) ---
    selected_candidate_texts: List[str] = None
    final_prompt_text: Optional[str] = None
    zero_score_fallback_triggered: bool = False
    executions: List[Dict[str, Any]] = None  
    pass_at_k_metrics: Dict[int, float] = None
    
    # === NEW: RANKING METRICS (BLOCK A ONLY) ===
    ap_score_reranked: Optional[float] = None  # AP on reranked list
    ap_score_original: Optional[float] = None  # AP on original retrieved list
    ap_improvement: Optional[float] = None  # Delta: reranked - original
    candidate_coverage_rate: Optional[float] = None  # Fraction of candidate set used
    avg_rerank_position_shift: Optional[float] = None  # Avg rank movement (Block A only)
    
    # Meta
    timestamp: str = None
    notes: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.selected_candidates is None:
            self.selected_candidates = []
        if self.selected_evaluators is None:
            self.selected_evaluators = []
        if self.selected_scores is None:
            self.selected_scores = []  # NEW
        if self.selected_exemplar_ids is None:
            self.selected_exemplar_ids = []
        if self.selected_exemplar_texts is None:
            self.selected_exemplar_texts = []
        if self.selected_candidate_texts is None:
            self.selected_candidate_texts = []
        if self.executions is None:
            self.executions = []
        if self.pass_at_k_metrics is None:
            self.pass_at_k_metrics = {}


@dataclass
class Layer2Config:
    """Configuration for Layer 2 experiments."""
    layer2_config_name: str = "default_run"  # <--- ADD THIS LINE
    
    # Block Execution Toggles
    run_block_A_baseline: bool = True  # <--- ADD THIS NEW LINE
    run_block_A: bool = True
    run_block_B: bool = True
    run_block_C: bool = True
    
    # Global Base Conditions
    utility_calibration_modes: List[str] = None  # NEW: ['Marginal', 'Absolute']
    evaluator_masking: List[str] = None  # ['Self', 'Others', 'All']
    base_scoring_strategies: List[str] = None  # ['ScoreTake', 'ScoreMake', 'Holistic']
    global_pass_at_N: int = 3
    activation_threshold: float = 0.0  # Threshold for positive signal
    
    # Block A Configurations
    block_A_strategies: List[str] = None  # Which strategies to use
    block_A_weight_taker: float = 1.0
    block_A_weight_maker: float = 1.0
    top_ks_group: List[int] = None  # [1, 3, 5]
    
    # Block B Configurations
    dynamic_k_methods: List[str] = None  # ['K_take', 'K_make', 'K_both']
    block_B_weight_taker: float = 1.0
    block_B_weight_maker: float = 1.0
    run_boundary_intersection_test: bool = False
    
    # Block C Configurations
    coverage_perspectives: List[str] = None  # ['Candidate_Centric', 'Evaluator_Centric']
    block_C_tiebreaker_weight_taker: float = 1.0
    block_C_tiebreaker_weight_maker: float = 1.0
    
    def __post_init__(self):
        if self.utility_calibration_modes is None:
            self.utility_calibration_modes = ['Marginal', 'Absolute']
        if self.evaluator_masking is None:
            self.evaluator_masking = ['Self', 'Others', 'All']
        if self.base_scoring_strategies is None:
            self.base_scoring_strategies = ['ScoreTake', 'ScoreMake', 'Holistic']
        if self.block_A_strategies is None:
            self.block_A_strategies = ['ScoreTake', 'ScoreMake', 'Holistic']
        if self.top_ks_group is None:
            self.top_ks_group = [1, 3, 5]
        if self.dynamic_k_methods is None:
            self.dynamic_k_methods = ['K_take', 'K_make', 'K_both']
        if self.coverage_perspectives is None:
            self.coverage_perspectives = ['Candidate_Centric', 'Evaluator_Centric']


# ============================================================================
# RANKING METRICS CALCULATOR (NEW)
# ============================================================================

class RankingMetricsCalculator:
    """
    Computes ranking-based metrics for evaluating retrieved and reranked lists.
    Metrics include: Average Precision (AP).
    """
    
    @staticmethod
    def calculate_ap(ranked_candidates: List[str], ground_truth_labels: Dict[str, bool]) -> float:
        """
        Calculate Average Precision (AP) for a ranked list.
        
        Args:
            ranked_candidates: List of candidate IDs in ranked order
            ground_truth_labels: Dict mapping candidate_id -> is_relevant (bool)
        
        Returns:
            AP score in [0.0, 1.0]
        """
        if not ranked_candidates or not ground_truth_labels:
            return 0.0
        
        # Count total relevant items
        num_relevant = sum(1 for label in ground_truth_labels.values() if label)
        if num_relevant == 0:
            return 0.0
        
        # Calculate precision at each relevant position
        ap_sum = 0.0
        num_relevant_found = 0
        
        for position, candidate_id in enumerate(ranked_candidates, start=1):
            is_relevant = ground_truth_labels.get(candidate_id, False)
            if is_relevant:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / position
                ap_sum += precision_at_k
        
        # AP = sum of precisions / total relevant items
        ap = ap_sum / num_relevant if num_relevant > 0 else 0.0
        return min(ap, 1.0)  # Ensure bounded to [0, 1]
    
    @staticmethod
    def compute_all_metrics(
        ranked_candidates_reranked: List[str],
        ranked_candidates_original: List[str],
        ground_truth_labels: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Compute all ranking metrics for both reranked and original lists.
        
        Args:
            ranked_candidates_reranked: Reranked candidate list
            ranked_candidates_original: Original (pre-reranking) candidate list
            ground_truth_labels: Dict mapping candidate_id -> is_relevant (bool)
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {
            'ap_reranked': RankingMetricsCalculator.calculate_ap(ranked_candidates_reranked, ground_truth_labels),
            'ap_original': RankingMetricsCalculator.calculate_ap(ranked_candidates_original, ground_truth_labels)
        }
        
        # Calculate AP improvement
        metrics['ap_improvement'] = metrics['ap_reranked'] - metrics['ap_original']
        
        return metrics


# ============================================================================
# PTU MATH ENGINE
# ============================================================================

class PTUMathEngine:
    """
    Computes Pairwise Transaction Utility (PTU) matrices and score aggregations
    from Layer 1 cached data.
    """
    
    def __init__(self, layer1_state: Dict[str, Any], exemplar_data: Dict[str, Any] = None, hard_questions: List[str] = None):
        """
        Initialize with Layer 1 state (from cache).
        
        Args:
            layer1_state: Complete Layer 1 cached state containing:
                - retrieved_set: List of retrieved exemplars
                - candidate_set: List of candidates with parent mapping
                - intrinsic_baselines: Dict of baseline success rates per evaluator
                - cross_evaluation_matrix: Raw binary success data
                - ground_truth_labels: True/False labels for candidates
        
        Raises:
            ValueError: If layer1_state is invalid or contains required empty structures
            RuntimeError: If PTU matrix dimensions do not match expected sizes
        """
        # === BUG FIX 3: STRICT LAYER 1 STATE VALIDATION ===
        # Validate that layer1_state is provided and is a dictionary
        if layer1_state is None:
            raise ValueError(
                "PTUMathEngine initialization failed: layer1_state is None. "
                "Layer 1 cache must be loaded before initializing the math engine."
            )
        if not isinstance(layer1_state, dict):
            raise ValueError(
                f"PTUMathEngine initialization failed: layer1_state must be a dictionary. "
                f"Got type: {type(layer1_state)}"
            )
        
        self.layer1_state = layer1_state
        self.exemplar_data = exemplar_data or {}
        self.hard_questions = hard_questions or []
        
        # =========================================================
        # BUG 1 FIX: BULLETPROOF INDEX EXTRACTION
        # =========================================================
        idx = layer1_state.get('target_query_idx') # 1. Try root
        
        if idx is None and 'target_query_data' in layer1_state:
            idx = layer1_state['target_query_data'].get('query_index') # 2. Try nested dict
            
        if idx is None: # 3. Failsafe: Text matching in RAM
            text_to_find = layer1_state.get('target_query_text')
            if not text_to_find and 'target_query_data' in layer1_state:
                text_to_find = layer1_state['target_query_data'].get('query_text')
            
            if text_to_find and text_to_find in self.hard_questions:
                idx = self.hard_questions.index(text_to_find)
                
        if idx is None:
            raise ValueError("Fatal: Could not determine target_query_idx from Layer 1 cache.")
            
        self.target_query_idx = int(idx)
        
        # === FETCH TEXT DIRECTLY FROM RAM ===
        self.target_query_text = self.hard_questions[self.target_query_idx]
        
        # Safely fetch ground truth
        if 'ground_truths' in self.exemplar_data:
            self.ground_truth_answer = self.exemplar_data['ground_truths'][self.target_query_idx]
        else:
            self.ground_truth_answer = self.exemplar_data.get('solutions', [])[self.target_query_idx]
        
        # Extract raw Layer 1 data structures
        self.raw_retrieved_set = layer1_state.get('retrieved_set', [])
        self.raw_candidate_set = layer1_state.get('candidate_set', {})
        self.intrinsic_baselines = layer1_state.get('intrinsic_baselines', {})
        self.cross_eval_matrix = layer1_state.get('cross_evaluation_matrix', {})
        self.ground_truth_labels = layer1_state.get('ground_truth_labels', {})
        
        # Normalize Layer 1 IDs to contiguous matrix indices
        self.evaluator_ids, self.retrieved_set = self._normalize_retrieved_set(self.raw_retrieved_set)
        self.candidate_ids, self.candidate_set = self._normalize_candidate_set(self.raw_candidate_set)
        
        # === BUG FIX 3: STRICT DIMENSIONAL VALIDATION AFTER NORMALIZATION ===
        # Validate that normalization produced non-empty structures
        if not self.evaluator_ids or len(self.evaluator_ids) == 0:
            raise ValueError(
                "PTUMathEngine initialization failed: Retrieved exemplar set is empty after normalization. "
                "Layer 1 cache must contain non-empty 'retrieved_set' with at least one evaluator."
            )
        
        if not self.candidate_ids or len(self.candidate_ids) == 0:
            raise ValueError(
                "PTUMathEngine initialization failed: Candidate set is empty after normalization. "
                "Layer 1 cache must contain non-empty 'candidate_set' with at least one candidate."
            )
        
        self.candidate_id_to_idx = {cid: idx for idx, cid in enumerate(self.candidate_ids)}
        self.evaluator_id_to_idx = {eid: idx for idx, eid in enumerate(self.evaluator_ids)}
        
        # Derived data
        self.n_candidates = len(self.candidate_set)
        self.n_evaluators = len(self.retrieved_set)
        
        # === BUG FIX 3: VALIDATE DIMENSIONS BEFORE PTU COMPUTATION ===
        # Ensure consistency between normalized IDs and data structures
        if self.n_candidates != len(self.candidate_ids):
            raise RuntimeError(
                f"PTUMathEngine initialization failed: Dimension mismatch. "
                f"n_candidates={self.n_candidates} but len(candidate_ids)={len(self.candidate_ids)}. "
                f"This indicates a critical inconsistency in candidate normalization."
            )
        
        if self.n_evaluators != len(self.evaluator_ids):
            raise RuntimeError(
                f"PTUMathEngine initialization failed: Dimension mismatch. "
                f"n_evaluators={self.n_evaluators} but len(evaluator_ids)={len(self.evaluator_ids)}. "
                f"This indicates a critical inconsistency in evaluator normalization."
            )
        
        # Compute core PTU matrix (defaults to Marginal for fallback)
        self.ptu_matrix = self.get_calibrated_ptu_matrix('Marginal')
        
        # Cache for computed scores
        self._score_cache = {}
        
        # We no longer need the trace hack, we fetch directly from RAM now.
        
        logger.info(
            f"PTU Math Engine initialized: "
            f"Query #{self.target_query_idx}, "
            f"{self.n_candidates} candidates, "
            f"{self.n_evaluators} evaluators"
        )
    
    def _normalize_candidate_set(self, candidate_set: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Normalize candidate data into an ordered list and ID mapping."""
        ids = []
        candidates = []

        if isinstance(candidate_set, dict):
            items = list(candidate_set.items())
            # Dictionary Sorting Bug Removed: Preserve true Layer 1 retrieval insertion order

            for key, value in items:
                candidate_id = None
                if isinstance(value, dict):
                    candidate_id = value.get('candidate_id')
                    if candidate_id is None:
                        candidate_id = value.get('source_exemplar_idx')
                if candidate_id is None:
                    candidate_id = key
                ids.append(str(candidate_id))
                candidates.append(value)

            return ids, candidates

        elif isinstance(candidate_set, list):
            for idx, value in enumerate(candidate_set):
                candidate_id = None
                if isinstance(value, dict):
                    candidate_id = value.get('candidate_id') or value.get('source_exemplar_idx')
                ids.append(str(candidate_id) if candidate_id is not None else str(idx))
                candidates.append(value)
            return ids, candidates

        return [], []
    
    def _normalize_retrieved_set(self, retrieved_set: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Normalize retrieved evaluator data into an ordered list and ID mapping."""
        ids = []
        evaluators = []

        if isinstance(retrieved_set, dict):
            items = list(retrieved_set.items())
            # Dictionary Sorting Bug Removed: Preserve true Layer 1 retrieval insertion order

            for key, value in items:
                evaluator_id = None
                if isinstance(value, dict):
                    evaluator_id = value.get('corpus_index') or value.get('retrieval_index') or value.get('retrieved_idx')
                if evaluator_id is None:
                    evaluator_id = key
                ids.append(str(evaluator_id))
                evaluators.append(value)

            return ids, evaluators

        elif isinstance(retrieved_set, list):
            for idx, item in enumerate(retrieved_set):
                evaluator_id = None
                if isinstance(item, dict):
                    evaluator_id = item.get('corpus_index') or item.get('retrieval_index') or item.get('retrieved_idx')
                ids.append(str(evaluator_id) if evaluator_id is not None else str(idx))
                evaluators.append(item)
            return ids, evaluators

        return [], []
    
    def _normalize_id(self, raw_id: Any) -> Any:
        """Normalize an ID to its string or integer representation."""
        if isinstance(raw_id, str):
            return raw_id
        if isinstance(raw_id, (int, np.integer)):
            return str(raw_id)
        return str(raw_id)
    
    def _lookup_value_by_key_variants(self, data: dict, key: Any, default: Any = None) -> Any:
        """Lookup a value by int/key/string variants to support Layer 1 state shapes."""
        if data is None:
            return default
        if key in data:
            return data[key]
        key_str = self._normalize_id(key)
        if key_str in data:
            return data[key_str]
        try:
            key_int = int(key)
            if key_int in data:
                return data[key_int]
        except Exception:
            pass
        return default
    
    def _fetch_cross_eval_score(self, candidate_id: Any, evaluator_id: Any) -> float:
        """Fetch a cross-evaluation score from Layer 1 state strictly. Fail on missing data."""
        row = self._lookup_value_by_key_variants(self.cross_eval_matrix, candidate_id, None)
        if not isinstance(row, dict):
            tuple_key = (candidate_id, evaluator_id)
            score = self._lookup_value_by_key_variants(self.cross_eval_matrix, tuple_key, None)
            if score is None:
                score = self._lookup_value_by_key_variants(self.cross_eval_matrix, str(tuple_key), None)
        else:
            score = self._lookup_value_by_key_variants(row, evaluator_id, None)
            
        if score is None:
            raise ValueError(f"Fatal API missing-data error: No cross-eval score found for candidate {candidate_id} and evaluator {evaluator_id}.")
        return float(score)
    
    def _fetch_intrinsic_baseline(self, evaluator_id: Any) -> float:
        """Fetch intrinsic baseline score from Layer 1 state strictly. Fail on missing data."""
        score = self._lookup_value_by_key_variants(self.intrinsic_baselines, evaluator_id, None)
        if score is None:
            raise ValueError(f"Fatal API missing-data error: No intrinsic baseline found for evaluator {evaluator_id}.")
        return float(score)
    
    def get_calibrated_ptu_matrix(self, calibration_mode: str = 'Marginal') -> np.ndarray:
        """
        Compute the base PTU matrix without masking, applying the requested mathematical calibration.
        
        Returns:
            Matrix of shape (n_candidates, n_evaluators) with PTU values.
        
        Raises:
            RuntimeError: If dimensional validation fails
        """
        # === BUG FIX 3: STRICT DIMENSIONAL VALIDATION BEFORE COMPUTATION ===
        # Validate that n_candidates and n_evaluators are properly set and consistent
        if self.n_candidates <= 0:
            raise RuntimeError(
                f"PTU matrix computation failed: Invalid n_candidates={self.n_candidates}. "
                f"The candidate set must contain at least 1 element."
            )
        
        if self.n_evaluators <= 0:
            raise RuntimeError(
                f"PTU matrix computation failed: Invalid n_evaluators={self.n_evaluators}. "
                f"The retrieved evaluator set must contain at least 1 element."
            )
        
        # Validate internal data structure consistency
        if len(self.candidate_set) != self.n_candidates:
            raise RuntimeError(
                f"PTU matrix computation failed: Candidate set size mismatch. "
                f"len(candidate_set)={len(self.candidate_set)} but n_candidates={self.n_candidates}. "
                f"This indicates a critical data structure inconsistency."
            )
        
        if len(self.retrieved_set) != self.n_evaluators:
            raise RuntimeError(
                f"PTU matrix computation failed: Evaluator set size mismatch. "
                f"len(retrieved_set)={len(self.retrieved_set)} but n_evaluators={self.n_evaluators}. "
                f"This indicates a critical data structure inconsistency."
            )
        
        # Validate ID mapping consistency
        if len(self.candidate_ids) != self.n_candidates:
            raise RuntimeError(
                f"PTU matrix computation failed: Candidate ID mapping size mismatch. "
                f"len(candidate_ids)={len(self.candidate_ids)} but n_candidates={self.n_candidates}."
            )
        
        if len(self.evaluator_ids) != self.n_evaluators:
            raise RuntimeError(
                f"PTU matrix computation failed: Evaluator ID mapping size mismatch. "
                f"len(evaluator_ids)={len(self.evaluator_ids)} but n_evaluators={self.n_evaluators}."
            )
        
        # === COMPUTE PTU MATRIX ===
        # Initialize the PTU matrix with dimensions (n_candidates, n_evaluators)
        ptu = np.zeros((self.n_candidates, self.n_evaluators), dtype=np.float32)
        
        # === BUG FIX 3: VALIDATE MATRIX DIMENSIONS AFTER CREATION ===
        if ptu.shape != (self.n_candidates, self.n_evaluators):
            raise RuntimeError(
                f"PTU matrix computation failed: Created matrix has invalid shape. "
                f"Expected ({self.n_candidates}, {self.n_evaluators}) but got {ptu.shape}."
            )
        
        # Populate the matrix
        for cand_idx, candidate in enumerate(self.candidate_set):
            if cand_idx >= self.n_candidates:
                raise RuntimeError(
                    f"PTU matrix computation failed: Candidate index {cand_idx} exceeds "
                    f"expected size {self.n_candidates}."
                )
            
            cand_id = self.candidate_ids[cand_idx]
            for eval_idx, evaluator in enumerate(self.retrieved_set):
                if eval_idx >= self.n_evaluators:
                    raise RuntimeError(
                        f"PTU matrix computation failed: Evaluator index {eval_idx} exceeds "
                        f"expected size {self.n_evaluators}."
                    )
                
                eval_id = self.evaluator_ids[eval_idx]
                induced_ccs = self._fetch_cross_eval_score(cand_id, eval_id)
                intrinsic_ccs = self._fetch_intrinsic_baseline(eval_id)
                
                # Validate fetched scores are numeric (Strict Validation)
                if not isinstance(induced_ccs, (int, float)) or np.isnan(induced_ccs):
                    raise ValueError(
                        f"Fatal math error: Invalid induced_ccs type/value for candidate {cand_id} "
                        f"and evaluator {eval_id}. Value: {induced_ccs}"
                    )
                
                if not isinstance(intrinsic_ccs, (int, float)) or np.isnan(intrinsic_ccs):
                    raise ValueError(
                        f"Fatal math error: Invalid intrinsic_ccs type/value for evaluator {eval_id}. "
                        f"Value: {intrinsic_ccs}"
                    )
                
                # NEW: Apply the mathematical toggle
                if calibration_mode == 'Absolute':
                    ptu[cand_idx, eval_idx] = float(induced_ccs)
                else:  # 'Marginal'
                    ptu[cand_idx, eval_idx] = max(0.0, induced_ccs - intrinsic_ccs)
        
        # Final validation: Ensure matrix is fully populated
        if np.all(ptu == 0.0):
            logger.warning(
                "PTU matrix computation completed, but all values are zero. "
                "This may indicate empty or mismatched cross-evaluation data in Layer 1 cache."
            )
        
        return ptu
    
    def apply_evaluator_mask(self, mask_type: str, base_matrix: np.ndarray) -> np.ndarray:
        """
        Apply evaluator masking to the provided PTU matrix.
        
        Args:
            mask_type: 'Self' (diagonal only), 'Others' (off-diagonal), or 'All' (no mask)
            base_matrix: The mathematically calibrated base PTU matrix
        
        Returns:
            Masked PTU matrix
        """
        # === BUG FIX 3: VALIDATE MASK APPLICATION ===
        # Validate mask_type
        valid_masks = ['Self', 'Others', 'All']
        if mask_type not in valid_masks:
            raise ValueError(
                f"apply_evaluator_mask failed: Invalid mask_type '{mask_type}'. "
                f"Must be one of {valid_masks}."
            )
        
        # Validate PTU matrix state before masking
        if base_matrix is None:
            raise RuntimeError(
                "apply_evaluator_mask failed: base_matrix is None. "
            )
        
        if base_matrix.shape != (self.n_candidates, self.n_evaluators):
            raise RuntimeError(
                f"apply_evaluator_mask failed: PTU matrix shape mismatch. "
                f"Expected ({self.n_candidates}, {self.n_evaluators}) "
                f"but got {base_matrix.shape}."
            )
        
        # Create a copy to avoid modifying the original
        masked_ptu = base_matrix.copy()
        
        if mask_type == 'Self':
            # Keep only the true parent-child evaluator relationships for each candidate.
            for i in range(self.n_candidates):
                candidate = self.candidate_set[i]
                true_j = self._resolve_source_evaluator_index(candidate, i)
                for j in range(self.n_evaluators):
                    if j != true_j:
                        masked_ptu[i, j] = 0.0
        
        elif mask_type == 'Others':
            # Keep everything except the true parent-child relationships.
            for i in range(self.n_candidates):
                candidate = self.candidate_set[i]
                true_j = self._resolve_source_evaluator_index(candidate, i)
                if 0 <= true_j < self.n_evaluators:
                    masked_ptu[i, true_j] = 0.0
        
        # 'All' leaves the matrix unchanged
        
        return masked_ptu
    
    def compute_score_take(self, ptu_matrix: np.ndarray) -> np.ndarray:
        """
        Compute ScoreTake: sum of PTU values for each candidate (row sums).
        
        Returns:
            Array of shape (n_candidates,) with score take values
        """
        return np.sum(ptu_matrix, axis=1)
    
    def compute_score_make(self, ptu_matrix: np.ndarray) -> np.ndarray:
        """
        Compute ScoreMake: sum of PTU values for each evaluator (column sums).
        
        Returns:
            Array of shape (n_evaluators,) with score make values
        """
        return np.sum(ptu_matrix, axis=0)
    
    def compute_holistic_score(
        self,
        ptu_matrix: np.ndarray,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> np.ndarray:
        """
        Compute Holistic Score: weighted combination of ScoreTake and ScoreMake.
        
        For each candidate i (which comes from evaluator src(i)):
        HolisticScore(i) = ScoreTake(i) * weight_taker + ScoreMake(src(i)) * weight_maker
        
        Returns:
            Array of shape (n_candidates,) with holistic scores
        """
        score_take = self.compute_score_take(ptu_matrix)
        score_make = self.compute_score_make(ptu_matrix)
        
        holistic = np.zeros(self.n_candidates, dtype=np.float32)
        
        for cand_idx, candidate in enumerate(self.candidate_set):
            src_eval_idx = self._resolve_source_evaluator_index(candidate, cand_idx)
            make_score = score_make[src_eval_idx] if 0 <= src_eval_idx < len(score_make) else 0.0
            holistic[cand_idx] = (
                score_take[cand_idx] * weight_taker +
                make_score * weight_maker
            )
        
        return holistic
    
    def get_target_query_embedding_similarity(self) -> Dict[int, float]:
        """Return the cached target query similarity score strictly. Fail on missing data."""
        similarity_map: Dict[int, float] = {}
        for evaluator_idx, evaluator in enumerate(self.retrieved_set):
            if not isinstance(evaluator, dict) or 'similarity_score' not in evaluator or evaluator['similarity_score'] is None:
                raise ValueError(
                    f"Fatal API missing-data error: No embedding 'similarity_score' found for retrieved sample {evaluator_idx}."
                )
            similarity_map[evaluator_idx] = float(evaluator['similarity_score'])
        return similarity_map

    def _resolve_source_evaluator_index(self, candidate: Dict[str, Any], default_idx: int) -> int:
        """Resolve the matrix evaluator index for a candidate's source exemplar strictly."""
        source_id = candidate.get('source_exemplar_idx') if isinstance(candidate, dict) else None
        
        # STRICT DATA INTEGRITY: Crash if the candidate has no parent assigned
        if source_id is None:
            raise ValueError(f"Fatal structural error: Candidate is missing 'source_exemplar_idx'. Data: {candidate}")
            
        source_key = self._normalize_id(source_id)
        
        # Crash if the parent ID doesn't actually exist in the retrieved set
        if source_key not in self.evaluator_id_to_idx:
            raise ValueError(f"Fatal structural error: Source ID '{source_key}' not found in retrieved evaluators list.")
            
        return self.evaluator_id_to_idx[source_key]
    
    def get_scores_for_strategy(
        self,
        ptu_matrix: np.ndarray,
        strategy: str,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> np.ndarray:
        """Get scores based on the specified strategy."""
        if strategy == 'ScoreTake':
            return self.compute_score_take(ptu_matrix)
        elif strategy == 'ScoreMake':
            score_make = self.compute_score_make(ptu_matrix)
            result = np.zeros(self.n_candidates, dtype=np.float32)
            for cand_idx, candidate in enumerate(self.candidate_set):
                src_eval_idx = self._resolve_source_evaluator_index(candidate, cand_idx)
                result[cand_idx] = score_make[src_eval_idx] if 0 <= src_eval_idx < len(score_make) else 0.0
            return result
        elif strategy == 'Holistic':
            return self.compute_holistic_score(ptu_matrix, weight_taker, weight_maker)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def candidate_indices_to_ids(self, candidate_indices: List[int]) -> List[str]:
        """Convert a list of internal candidate indices into dataset IDs."""
        output_ids = []
        for idx in candidate_indices:
            if 0 <= idx < len(self.candidate_set):
                candidate = self.candidate_set[idx]
                explicit_id = None
                if isinstance(candidate, dict):
                    explicit_id = candidate.get('candidate_id') or candidate.get('source_exemplar_idx')
                if explicit_id is not None:
                    output_ids.append(str(explicit_id))
                else:
                    output_ids.append(self.candidate_ids[idx])
        return output_ids

    def evaluator_indices_to_ids(self, evaluator_indices: List[int]) -> List[str]:
        """Convert a list of internal evaluator indices into dataset IDs."""
        return [self.evaluator_ids[idx] for idx in evaluator_indices if 0 <= idx < len(self.evaluator_ids)]

    def get_candidate_texts(self, candidate_indices: List[int]) -> List[str]:
        """Extract the actual text strings for the selected candidates."""
        texts = []
        for idx in candidate_indices:
            if 0 <= idx < len(self.candidate_set):
                cand = self.candidate_set[idx]
                if isinstance(cand, dict) and cand.get('candidate_text'):
                    texts.append(cand['candidate_text'])
        return texts


    def get_source_exemplars(self, candidate_indices: List[int]) -> Tuple[List[str], List[str]]:
        """
        Map candidate indices to their parent Source Exemplar IDs and Texts (Q+A).
        Deduplicates results and ignores Zero-Shot candidates (-1).
        Fetches text securely from the raw exemplar_data in memory.
        """
        seen_ids = set()
        ids = []
        texts = []
        
        for cand_idx in candidate_indices:
            if 0 <= cand_idx < len(self.candidate_set):
                cand = self.candidate_set[cand_idx]
                src_eval_idx = self._resolve_source_evaluator_index(cand, cand_idx)
                
                if 0 <= src_eval_idx < len(self.evaluator_ids):
                    eval_id = self.evaluator_ids[src_eval_idx]
                    
                    # Ignore Zero-Shot and duplicates
                    if eval_id == "-1" or eval_id in seen_ids:
                        continue
                        
                    seen_ids.add(eval_id)
                    ids.append(eval_id)
                    
                    # === DIRECT MEMORY LOOKUP (Bug 2 Fixed) ===
                    try:
                        raw_index = int(eval_id)
                        q = self.exemplar_data['questions'][raw_index]
                        a = self.exemplar_data['solutions'][raw_index]
                        texts.append(f"Question: {q}\nRationale and Answer: {a}")
                    except (ValueError, TypeError, IndexError) as e:
                        raise ValueError(
                            f"Fatal error mapping eval_id '{eval_id}' to dataset. "
                            f"Make sure exemplar_data was passed correctly. Error: {e}"
                        )
                        
        return ids, texts
    
    def get_original_retrieved_ranking(self) -> List[str]:
        """
        Get the original (pre-reranking) candidate ranking from Layer 1 retrieved set.
        Candidates are ordered by their retrieval index (as they appear in the retrieved set).
        
        Returns:
            List of candidate IDs in original retrieval order
        """
        # The original ranking is determined by the order of candidates in the retrieved set
        # (i.e., the order they were retrieved by the embedding model)
        # For now, we use the order of candidate_ids (which reflects the candidate_set order)
        # In Layer 1, candidates are typically ordered by their appearance/importance in retrievals
        return self.candidate_ids.copy()
    
    def compute_position_shift(self, original_indices: List[int], reranked_indices: List[int]) -> float:
        """
        Calculate average position shift from original to reranked ranking.
        
        Args:
            original_indices: Original candidate indices in order
            reranked_indices: Reranked candidate indices in order
        
        Returns:
            Average absolute position shift
        """
        if not reranked_indices:
            return 0.0
        
        # Map original candidates to their original positions
        original_positions = {cand_idx: pos for pos, cand_idx in enumerate(original_indices)}
        
        # Calculate position shift for reranked candidates
        total_shift = 0.0
        for new_pos, cand_idx in enumerate(reranked_indices):
            original_pos = original_positions.get(cand_idx, len(original_indices))
            shift = abs(original_pos - new_pos)
            total_shift += shift
        
        avg_shift = total_shift / len(reranked_indices) if reranked_indices else 0.0
        return avg_shift


def _normalize_ground_truth_label(label_obj: Any) -> bool:
    """Normalize a ground truth label value into a boolean is_correct flag."""
    if isinstance(label_obj, dict):
        return bool(label_obj.get('is_correct', False))
    return bool(label_obj)


def _get_ground_truth_label(ground_truth_labels: Dict[Any, Any], cand_idx: Any) -> bool:
    """Fetch and normalize a candidate's ground truth correctness label strictly."""
    if cand_idx in ground_truth_labels:
        return _normalize_ground_truth_label(ground_truth_labels[cand_idx])
    cand_key = str(cand_idx)
    if cand_key in ground_truth_labels:
        return _normalize_ground_truth_label(ground_truth_labels[cand_key])
    
    # STRICT DATA INTEGRITY: Crash if the evaluator API failed to record a label
    raise ValueError(f"Fatal API missing-data error: No ground truth label found for candidate index {cand_idx}.")


# ============================================================================
# ACTIVE INFERENCE ENGINE (LIVE LLM EVALUATION)
# ============================================================================

class ActiveInferenceEngine:
    """
    Executes live API calls to the LLM using the contexts chosen by Layer 2 Blocks.
    """
    def __init__(self, api_manager_solve: Any, api_manager_eval: Any, global_config: Optional[Dict[str, Any]] = None):
        self.api_manager_solve = api_manager_solve
        self.api_manager_eval = api_manager_eval
        self.global_config = global_config if global_config is not None else GLOBAL_CONFIG
        
        # Determine which model to use based on the API Manager type
        if isinstance(self.api_manager_solve, GeminiAPIManager):
            self.model_name = self.global_config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
        elif isinstance(self.api_manager_solve, AvalAIAPIManager):
            self.model_name = self.global_config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
        else:
            self.model_name = self.global_config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')

    def execute_and_evaluate(self, target_query: str, ground_truth: str, context_texts: List[str], n_attempts: int):
        # NEW: Deduplicate text strings to save tokens and prevent redundant context
        unique_contexts = []
        if context_texts:
            for text in context_texts:
                if text not in unique_contexts:
                    unique_contexts.append(text)
            context_texts = unique_contexts

        # 1. Prompt Assembly
        if not context_texts:
            prompt = create_final_reasoning_prompt_simple(target_query, self.global_config)
        else:
            formatted_contexts = []
            for text in context_texts:
                if "Question:" in text and "Rationale and Answer:" in text:
                    formatted_contexts.append(text)
                else:
                    formatted_contexts.append(f"Question: {target_query}\nRationale and Answer: {text}")
            prompt = create_final_reasoning_prompt(target_query, formatted_contexts, self.global_config)

        executions = []
        temp = self.global_config.get("DEFAULT_PASS_N_SOLVER_TEMPERATURE", 1.0)

        # 2. Worker function with RETRY LOGIC (Exponential Backoff)
        def run_attempt(i):
            max_retries = 3
            backoff = 2.0
            
            for attempt in range(max_retries):
                resp = self.api_manager_solve.generate_content(prompt, self.model_name, temp)
                raw_text = resp.get('text', '')
                error_msg = resp.get('error_message', '')
                
                if resp['status'] == 'SUCCESS':
                    eval_res = evaluate_single_answer_with_llm(raw_text, ground_truth, self.api_manager_eval, self.global_config)
                    
                    # NEW: Did the evaluator API fail? If yes, trigger a retry!
                    if eval_res.get('status') != 'SUCCESS':
                        time.sleep(backoff)
                        backoff *= 2
                        continue 
                        
                    return {
                        "attempt_index": i + 1,
                        "raw_llm_generation": raw_text,
                        "error": "",
                        "is_correct": eval_res.get('is_correct', False),
                        "api_success": True
                    }
                else:
                    time.sleep(backoff)
                    backoff *= 2  # Double the wait time
            
            # If all retries fail
            return {
                "attempt_index": i + 1,
                "raw_llm_generation": "",
                "error": error_msg,
                "is_correct": False,
                "api_success": False  # Mark as API failure so we don't count it in the math
            }

        # 3. Execute N Independent Inferences Sequentially (One by one)
        for i in range(n_attempts):
            # Print a tiny log so you know it's working sequentially
            print(f"      -> Running LLM inference attempt {i+1} of {n_attempts}...")
            
            # Run the attempt and wait for it to finish before moving to the next
            result = run_attempt(i)
            executions.append(result)
        
        executions.sort(key=lambda x: x["attempt_index"])

        # 4. Calculate empirical Pass@k metrics (Excluding API Failures)
        pass_at_k = {}
        # Only count successful API calls for our math
        valid_executions = [ex for ex in executions if ex["api_success"]]
        is_correct_array = [ex["is_correct"] for ex in valid_executions]
        
        # If all API calls failed, return 0.0 for everything
        actual_n = len(valid_executions)
        for k in range(1, n_attempts + 1):
            if k <= actual_n:
                pass_at_k[k] = 1.0 if any(is_correct_array[:k]) else 0.0
            else:
                pass_at_k[k] = pass_at_k.get(actual_n, 0.0) # Carry forward the last valid score

        return prompt, executions, pass_at_k


# ============================================================================
# BLOCK A: BASELINE RERANKING & STATIC GROUPING
# ============================================================================

class BlockA:
    """
    Baseline application: Reranking by scores and static Top-K slicing.
    """
    
    def __init__(self, ptu_engine: PTUMathEngine, config: Layer2Config):
        self.ptu_engine = ptu_engine
        self.config = config
        self.results = []
        # Store ranked indices for each (mask_type, strategy) combination for Block B boundary test
        self.ranked_indices_cache = {}
    

    def run_original_baseline(self) -> List[ExperimentResult]:
        """
        Runs the Active Baseline for the Original (Un-reranked) Retrieval List.
        Creates ExperimentResult objects for Top-K slicing of the original list.
        """
        results = []
        # Get the original sequence (0, 1, 2, 3...)
        original_indices = list(range(len(self.ptu_engine.candidate_ids)))
        original_ranking_ids = self.ptu_engine.get_original_retrieved_ranking()
        
        ground_truth_labels_dict = {
            # STRICT DATA INTEGRITY: Use the safe function that crashes on missing labels
            self.ptu_engine.candidate_ids[i]: _get_ground_truth_label(
                self.ptu_engine.ground_truth_labels, 
                self.ptu_engine.candidate_ids[i]  # <--- BUG 3 FIX: Use the actual mapped ID, not the loop index 'i'
            )
            for i in range(len(self.ptu_engine.candidate_ids))
        }

        for k in self.config.top_ks_group:
            if k > len(original_indices):
                continue
                
            top_k_indices = original_indices[:k]
            top_k_candidate_ids = self.ptu_engine.candidate_indices_to_ids(top_k_indices)
            
            # Since this IS the original list, reranked metrics equal original metrics
            top_k_ranking_metrics = RankingMetricsCalculator.compute_all_metrics(
                top_k_candidate_ids,
                original_ranking_ids,
                ground_truth_labels_dict
            )
            
            cand_texts = self.ptu_engine.get_candidate_texts(top_k_indices)
            exemplar_ids, exemplar_texts = self.ptu_engine.get_source_exemplars(top_k_indices)
            
            result = ExperimentResult(
                target_query_idx=self.ptu_engine.target_query_idx,
                target_query_text=self.ptu_engine.target_query_text,
                ground_truth_answer=self.ptu_engine.ground_truth_answer,
                utility_calibration="Baseline",
                evaluator_setting="Baseline",
                scoring_strategy="Original_Retrieval",
                weight_taker=1.0,
                weight_maker=1.0,
                application=f"Block_A_Baseline_TopK_{k}",
                subset_size=k,
                selected_candidates=top_k_candidate_ids,
                selected_scores=[0.0 for _ in top_k_indices], # No PTU scores for baseline
                selected_exemplar_ids=exemplar_ids,
                selected_exemplar_texts=exemplar_texts,
                selected_candidate_texts=cand_texts,
                zero_score_fallback_triggered=(len(cand_texts) == 0),
                list_ap_score=None,
                group_pass_at_n=None,
                ap_score_reranked=top_k_ranking_metrics['ap_original'],
                ap_score_original=top_k_ranking_metrics['ap_original'],
                ap_improvement=0.0, # Baseline cannot improve upon itself
                candidate_coverage_rate=k / len(original_indices) if original_indices else 0.0,
                avg_rerank_position_shift=0.0
            )
            results.append(result)
            
            logger.info(
                f"Block A BASELINE - Original_Retrieval Top-{k}: "
                f"AP_Original = {top_k_ranking_metrics['ap_original']:.4f}, "
                f"Pass@{self.config.global_pass_at_N} = PENDING"
            )
            
        self.results.extend(results)
        return results

    def run_for_mask_and_strategy(
        self,
        ptu_matrix: np.ndarray,
        utility_calibration: str,
        mask_type: str,
        strategy: str,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> List[ExperimentResult]:
        """
        Run Block A experiments for a specific mask and strategy combination.
        
        Implements two-tier sorting system for Graceful Similarity Fallback:
        ...
        """
        results = []
        threshold = self.config.activation_threshold
        
        # ---> FIX: Fetch the original ranking IDs for the AP metrics calculator <---
        original_ranking_ids = self.ptu_engine.get_original_retrieved_ranking()
        
        # Get scores for this strategy
        scores = self.ptu_engine.get_scores_for_strategy(
            ptu_matrix, strategy, weight_taker, weight_maker
        )
        
        # Experiment A.1: Reranking & Average Precision with Two-Tier Sorting
        # ===== TWO-TIER SORTING SYSTEM =====
        # Tier 1: Candidates with score > threshold (sorted descending)
        tier1_mask = scores > threshold
        tier1_indices = np.where(tier1_mask)[0]
        tier1_sorted = tier1_indices[np.argsort(-scores[tier1_indices])]
        
        # Tier 2: Candidates with score <= threshold (sorted by original index - ascending)
        tier2_mask = scores <= threshold
        tier2_indices = np.where(tier2_mask)[0]
        tier2_sorted = np.sort(tier2_indices)  # Already in original order
        
        # Merge: Concatenate Tier 1 and Tier 2 to form complete ranked list
        ranked_indices = np.concatenate([tier1_sorted, tier2_sorted])
        
        # Verify list length equals total number of candidates
        assert len(ranked_indices) == len(scores), (
            f"Ranked list length mismatch: {len(ranked_indices)} != {len(scores)}"
        )
        
        # Cache the ranked indices for Block B boundary test (Experiment B.2.3)
        cache_key = (mask_type, strategy)
        self.ranked_indices_cache[cache_key] = ranked_indices.tolist()
        
        ranked_candidate_ids = self.ptu_engine.candidate_indices_to_ids(ranked_indices.tolist())
        
        ground_truth_labels_dict = {
            # STRICT DATA INTEGRITY: Use the safe function that crashes on missing labels
            self.ptu_engine.candidate_ids[i]: _get_ground_truth_label(
                self.ptu_engine.ground_truth_labels, 
                self.ptu_engine.candidate_ids[i]  # <--- BUG 3 FIX: Use the actual mapped ID, not the loop index 'i'
            )
            for i in range(len(self.ptu_engine.candidate_ids))
        }
        
        # Compute all ranking metrics
        ranking_metrics = RankingMetricsCalculator.compute_all_metrics(
            ranked_candidate_ids,
            original_ranking_ids,
            ground_truth_labels_dict
        )
        
        # Calculate position shift from original to reranked
        original_indices = [i for i in range(len(self.ptu_engine.candidate_ids))]
        avg_position_shift = self.ptu_engine.compute_position_shift(original_indices, ranked_indices.tolist())
        
        # Calculate candidate coverage rate
        total_candidates = len(self.ptu_engine.candidate_ids)
        coverage_rate = len(ranked_indices) / total_candidates if total_candidates > 0 else 0.0
        
        cand_texts_a1 = self.ptu_engine.get_candidate_texts(ranked_indices.tolist())
        exemplar_ids_a1, exemplar_texts_a1 = self.ptu_engine.get_source_exemplars(ranked_indices.tolist())
        
        result_a1 = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            utility_calibration=utility_calibration,
            evaluator_setting=mask_type,
            scoring_strategy=strategy,
            weight_taker=weight_taker,
            weight_maker=weight_maker,
            application=f"Block_A_Reranking_{strategy}",
            subset_size=len(ranked_indices),
            selected_candidates=ranked_candidate_ids,
            selected_scores=[float(scores[idx]) for idx in ranked_indices],
            selected_exemplar_ids=exemplar_ids_a1,
            selected_exemplar_texts=exemplar_texts_a1,
            selected_candidate_texts=cand_texts_a1,
            zero_score_fallback_triggered=(len(cand_texts_a1) == 0),
            list_ap_score=None,
            group_pass_at_n=None,
            # === NEW: Block A specific metrics ===
            ap_score_reranked=ranking_metrics['ap_reranked'],
            ap_score_original=ranking_metrics['ap_original'],
            ap_improvement=ranking_metrics['ap_improvement'],
            candidate_coverage_rate=coverage_rate,
            avg_rerank_position_shift=avg_position_shift
        )
        results.append(result_a1)
        
        logger.info(
            f"Block A A.1 - {mask_type} {strategy}: "
            f"AP_Reranked = {ranking_metrics['ap_reranked']:.4f}, "
            f"AP_Original = {ranking_metrics['ap_original']:.4f}, "
            f"AP_Improvement = {ranking_metrics['ap_improvement']:.4f}"
        )
        
        # Experiment A.2: Static Top-K Grouping
        for k in self.config.top_ks_group:
            if k > len(ranked_indices):
                continue
            
            top_k_indices = ranked_indices[:k].tolist()
            top_k_candidate_ids = self.ptu_engine.candidate_indices_to_ids(top_k_indices)
            pass_at_n = None
            
            # Compute ranking metrics for top-K list
            top_k_ranking_metrics = RankingMetricsCalculator.compute_all_metrics(
                top_k_candidate_ids,
                original_ranking_ids,
                ground_truth_labels_dict
            )
            
            # Calculate position shift for top-K
            top_k_position_shift = self.ptu_engine.compute_position_shift(original_indices, top_k_indices)
            
            # Coverage rate for top-K
            top_k_coverage = k / total_candidates if total_candidates > 0 else 0.0
            
            cand_texts_a2 = self.ptu_engine.get_candidate_texts(top_k_indices)
            exemplar_ids_a2, exemplar_texts_a2 = self.ptu_engine.get_source_exemplars(top_k_indices)
            
            result_a2 = ExperimentResult(
                target_query_idx=self.ptu_engine.target_query_idx,
                target_query_text=self.ptu_engine.target_query_text,
                ground_truth_answer=self.ptu_engine.ground_truth_answer,
                utility_calibration=utility_calibration,
                evaluator_setting=mask_type,
                scoring_strategy=strategy,
                weight_taker=weight_taker,
                weight_maker=weight_maker,
                application=f"Block_A_TopK_{k}_{strategy}",
                subset_size=k,
                selected_candidates=top_k_candidate_ids,
                selected_scores=[float(scores[idx]) for idx in top_k_indices],
                selected_exemplar_ids=exemplar_ids_a2,
                selected_exemplar_texts=exemplar_texts_a2,
                selected_candidate_texts=cand_texts_a2,
                zero_score_fallback_triggered=(len(cand_texts_a2) == 0),
                list_ap_score=None,
                group_pass_at_n=pass_at_n,
                # === NEW: Block A TopK metrics ===
                ap_score_reranked=top_k_ranking_metrics['ap_reranked'],
                ap_score_original=top_k_ranking_metrics['ap_original'],
                ap_improvement=top_k_ranking_metrics['ap_improvement'],
                candidate_coverage_rate=top_k_coverage,
                avg_rerank_position_shift=top_k_position_shift,
            )
            results.append(result_a2)
            
            logger.info(
                f"Block A A.2 - {mask_type} {strategy} Top-{k}: "
                f"AP_Reranked = {top_k_ranking_metrics['ap_reranked']:.4f}, "
                f"Pass@{self.config.global_pass_at_N} = PENDING"
            )
        
        self.results.extend(results)
        return results
    
    def get_ranked_indices_for_strategy(self, mask_type: str, strategy: str) -> Optional[List[int]]:
        """
        Retrieve the cached ranked indices for a specific mask_type and strategy.
        Used by Block B boundary test to use Block A's ranking instead of recalculating.
        """
        cache_key = (mask_type, strategy)
        return self.ranked_indices_cache.get(cache_key, None)


# BLOCK B: DYNAMIC SMART-K GROUPING
class BlockB:
    """
    Advanced application: Dynamic K sizing based on score thresholds.
    """
    
    def __init__(self, ptu_engine: PTUMathEngine, config: Layer2Config):
        self.ptu_engine = ptu_engine
        self.config = config
        self.results = []
    
    def run_for_mask_and_method(
        self,
        ptu_matrix: np.ndarray,
        utility_calibration: str,
        mask_type: str,
        method: str,
        ranked_list_for_boundary: Optional[List[int]] = None,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> List[ExperimentResult]:
        """
        Run Block B experiments for a specific mask and dynamic-K method.
        
        Implements Smart Fallback:
        - If positive_indices found: K_dynamic = len(positive_indices)
        - If all candidates <= threshold: K_dynamic = max(config.top_ks_group)
        """
        results = []
        threshold = self.config.activation_threshold
        
        if method == 'K_take':
            scores = self.ptu_engine.compute_score_take(ptu_matrix)
        elif method == 'K_make':
            scores = self.ptu_engine.compute_score_make(ptu_matrix)
        elif method == 'K_both':
            scores = self.ptu_engine.compute_holistic_score(
                ptu_matrix, weight_taker, weight_maker
            )
        else:
            raise ValueError(f"Unknown dynamic-K method: {method}")
        
        # Find candidates/evaluators with positive signals
        if method in ['K_take', 'K_both']:
            # For candidate-based methods, filter candidates
            positive_indices = np.where(scores > threshold)[0].tolist()
        else:
            # For evaluator-based, find candidates whose source evaluators score > threshold
            positive_indices = []
            for cand_idx, candidate in enumerate(self.ptu_engine.candidate_set):
                src_eval_idx = self.ptu_engine._resolve_source_evaluator_index(candidate, cand_idx)
                if 0 <= src_eval_idx < len(scores) and scores[src_eval_idx] > threshold:
                    positive_indices.append(cand_idx)
        
        # SMART FALLBACK LOGIC 
        zero_score_fallback_triggered = False
        if len(positive_indices) > 0:
            # Normal case: K_dynamic = count of positive indices
            k_dynamic = len(positive_indices)
        else:
            # Fallback: All candidates scored <= threshold
            # Use the maximum K from top_ks_group as fallback
            k_dynamic = max(self.config.top_ks_group) if self.config.top_ks_group else 1
            # Cap to not exceed total number of candidates
            k_dynamic = min(k_dynamic, len(self.ptu_engine.candidate_set))
            zero_score_fallback_triggered = True
            
            # Use first K_dynamic candidates from the reranked Block A list
            # (which naturally puts original retrieval order if all PTU scores are 0)
            if ranked_list_for_boundary:
                positive_indices = ranked_list_for_boundary[:k_dynamic]
            else:
                # Fallback to first K candidates in original order
                positive_indices = list(range(k_dynamic))
        
        # Evaluate the dynamic group
        selected_candidate_ids = self.ptu_engine.candidate_indices_to_ids(positive_indices)
        exemplar_ids_b, exemplar_texts_b = self.ptu_engine.get_source_exemplars(positive_indices)
        pass_at_n = None
        
        result = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            utility_calibration=utility_calibration,
            evaluator_setting=mask_type,
            scoring_strategy=method,
            weight_taker=weight_taker,
            weight_maker=weight_maker,
            application=f"Block_B_Dynamic_{method}",
            subset_size=k_dynamic,
            selected_candidates=selected_candidate_ids,
            selected_scores=[float(scores[idx]) for idx in positive_indices],  # NEW: Grab the scores
            selected_exemplar_ids=exemplar_ids_b,
            selected_exemplar_texts=exemplar_texts_b,
            selected_candidate_texts=self.ptu_engine.get_candidate_texts(positive_indices),
            zero_score_fallback_triggered=zero_score_fallback_triggered,
            list_ap_score=None,
            group_pass_at_n=pass_at_n
        )
        results.append(result)
        
        logger.info(
            f"Block B - {mask_type} {method}: "
            f"K_dynamic = {k_dynamic}, "
            f"Fallback_Triggered = {zero_score_fallback_triggered}, "
            f"Pass@{self.config.global_pass_at_N} = PENDING"
        )
        
        # Experiment B.2.3: Boundary Intersection Test (if enabled)
        if self.config.run_boundary_intersection_test and ranked_list_for_boundary:
            # Use k_dynamic as cutoff on the reranked list
            cutoff_list = ranked_list_for_boundary[:k_dynamic]
            
            # Calculate encapsulation accuracy (True labels inside, False outside)
            cutoff_candidate_ids = self.ptu_engine.candidate_indices_to_ids(cutoff_list)
            outside_candidate_ids = self.ptu_engine.candidate_indices_to_ids(ranked_list_for_boundary[k_dynamic:])
            true_inside = sum(
                1 for cand_id in cutoff_candidate_ids
                if _get_ground_truth_label(self.ptu_engine.ground_truth_labels, cand_id)
            )
            false_outside = sum(
                1 for cand_id in outside_candidate_ids
                if not _get_ground_truth_label(self.ptu_engine.ground_truth_labels, cand_id)
            )
            total_outside = len(ranked_list_for_boundary) - k_dynamic
            
            encapsulation = (true_inside + (false_outside if total_outside > 0 else 0)) / len(ranked_list_for_boundary)
            
            result.notes = f"Boundary encapsulation: {encapsulation:.4f}"
            logger.info(f"  Boundary encapsulation: {encapsulation:.4f}")
        
        self.results.append(result)
        return results


# BLOCK C: OPTIMAL SUBSET (COVERAGE) GROUPING
class BlockC:
    """
    Advanced application: Optimal subset using coverage-based peak finding.
    """
    
    def __init__(self, ptu_engine: PTUMathEngine, config: Layer2Config):
        self.ptu_engine = ptu_engine
        self.config = config
        self.results = []
    
    def _apply_hierarchical_tiebreaker_candidate_centric(
        self,
        tied_candidates: List[int],
        ptu_matrix: np.ndarray,
        eval_idx: int,
        score_take: np.ndarray,
        target_query_embedding_similarity: Dict[int, float],
        already_selected_candidates: Set[int],
        holistic_scores: Optional[np.ndarray] = None
    ) -> int:
        """
        Apply hierarchical tie-breaking for candidate-centric view.
        Levels: 1) Coverage overlap, 2) Highest ScoreTake, 3) Highest Holistic,
                4) Highest embedding similarity
        
        Args:
            holistic_scores: Pre-computed holistic scores array (shape: n_candidates,).
                            If None, will be computed on-demand (inefficient if called multiple times).
        """
        # Level 1: Maximize coverage overlap (prefer candidates already selected elsewhere)
        overlap = [c for c in tied_candidates if c in already_selected_candidates]
        current_pool = overlap if overlap else tied_candidates

        # Level 2: Highest total ScoreTake
        best_idx = max(current_pool, key=lambda i: score_take[i])
        if len(current_pool) > 1:
            max_score_take = score_take[best_idx]
            tied_by_score = [i for i in current_pool if score_take[i] == max_score_take]
            
            if len(tied_by_score) > 1:
                # Level 3: Highest Holistic Score
                if holistic_scores is None:
                    # Fallback: compute if not provided (should rarely happen in optimized path)
                    holistic_scores = self.ptu_engine.compute_holistic_score(
                        ptu_matrix,
                        self.config.block_C_tiebreaker_weight_taker,
                        self.config.block_C_tiebreaker_weight_maker
                    )
                best_idx = max(tied_by_score, key=lambda i: holistic_scores[i])
                
                # Level 4: Highest embedding similarity
                tied_by_holistic = [
                    i for i in tied_by_score
                    if holistic_scores[i] == holistic_scores[best_idx]
                ]
                if len(tied_by_holistic) > 1:
                    # FIX: Resolve candidate index to source evaluator index for correct domain lookup
                    best_idx = max(
                        tied_by_holistic,
                        key=lambda cand_idx: target_query_embedding_similarity.get(
                            self.ptu_engine._resolve_source_evaluator_index(
                                self.ptu_engine.candidate_set[cand_idx],
                                cand_idx
                            ),
                            0.0
                        )
                    )
        
        return best_idx
    
    def _apply_hierarchical_tiebreaker_evaluator_centric(
        self,
        tied_evaluators: List[int],
        ptu_matrix: np.ndarray,
        cand_idx: int,
        score_make: np.ndarray,
        target_query_embedding_similarity: Dict[int, float],
        evaluator_max_counts: Dict[int, int]
    ) -> int:
        """
        Apply hierarchical tie-breaking for evaluator-centric view.
        Levels: 1) Coverage overlap by global frequency, 2) Highest ScoreMake, 3) Embedding similarity
        """
        # Level 1: Choose the evaluator that appears most often as a row-max across all candidates.
        # This is a global frequency-based tie-breaker, not an order-dependent greedy selection.
        row = ptu_matrix[cand_idx, :]
        max_ptu = np.max(row)
        current_pool = tied_evaluators
        threshold = getattr(self.config, 'activation_threshold', 0.0)
        if max_ptu > threshold:
            counts_excluding_current = {
                e: evaluator_max_counts.get(e, 0) - (1 if e in tied_evaluators else 0)
                for e in tied_evaluators
            }
            best_count = max(counts_excluding_current.values())
            frequent_evaluators = [e for e, count in counts_excluding_current.items() if count == best_count]
            if frequent_evaluators:
                current_pool = frequent_evaluators

        # Level 2: Highest total ScoreMake
        best_idx = max(current_pool, key=lambda i: score_make[i])
        
        if len(current_pool) > 1:
            max_score_make = score_make[best_idx]
            tied_by_score = [i for i in current_pool if score_make[i] == max_score_make]
            
            if len(tied_by_score) > 1:
                # Level 3: Highest embedding similarity
                best_idx = max(
                    tied_by_score,
                    key=lambda i: target_query_embedding_similarity.get(i, 0.0)
                )
        
        return best_idx

    def _compute_evaluator_maximum_counts(
        self,
        ptu_matrix: np.ndarray,
        threshold: float
    ) -> Dict[int, int]:
        """Compute how many times each evaluator is a row-max across the candidate set."""
        counts: Dict[int, int] = defaultdict(int)
        for cand_idx in range(self.ptu_engine.n_candidates):
            row = ptu_matrix[cand_idx, :]
            max_ptu = np.max(row)
            if max_ptu > threshold:
                tied_evaluators = np.where(row == max_ptu)[0].tolist()
                for evaluator_idx in tied_evaluators:
                    counts[evaluator_idx] += 1
        return counts

    def run_for_mask_and_perspective(
        self,
        ptu_matrix: np.ndarray,
        utility_calibration: str,
        mask_type: str,
        perspective: str,
        target_query_embedding_similarity: Dict[int, float]
    ) -> List[ExperimentResult]:
        """
        Run Block C experiments for a specific mask and perspective.
        
        Implements Bypass & Fallback:
        - If np.max(ptu_matrix) <= threshold: bypass complex coverage logic
        - Fallback to first K_fallback candidates in original order
        """
        results = []
        threshold = self.config.activation_threshold
        
        # ===== CHECK FOR ZERO-SCORE FALLBACK CONDITION =====
        max_ptu_value = np.max(ptu_matrix) if ptu_matrix.size > 0 else 0.0
        zero_score_fallback_triggered = (max_ptu_value <= threshold)
        
        if zero_score_fallback_triggered:
            # ===== FALLBACK PATH: Bypass complex coverage logic =====
            # STRICT FALLBACK: If math is useless, return ALL members of the retrieved list
            k_fallback = len(self.ptu_engine.candidate_set)
            
            # Use first K_fallback indices (original retrieval order)
            selected_candidate_indices = list(range(k_fallback))
            subset_size = k_fallback
            
            logger.info(
                f"Block C - {mask_type} {perspective} (ZERO-SCORE FALLBACK): "
                f"Max PTU = {max_ptu_value:.4f} <= {threshold:.4f}, "
                f"Using first {k_fallback} candidates in original order"
            )
        else:
            # ===== NORMAL PATH: Execute coverage logic =====
            score_take = self.ptu_engine.compute_score_take(ptu_matrix)
            score_make = self.ptu_engine.compute_score_make(ptu_matrix)
            
            if perspective == 'Candidate_Centric':
                # OPTIMIZATION: Pre-compute holistic scores once for all tie-breaker calls
                holistic_scores_candidate_centric = self.ptu_engine.compute_holistic_score(
                    ptu_matrix,
                    self.config.block_C_tiebreaker_weight_taker,
                    self.config.block_C_tiebreaker_weight_maker
                )
                
                # Find max PTU for each evaluator (column maxima)
                winning_source_evals = set()
                
                already_selected_cands: Set[int] = set()
                for eval_idx in range(self.ptu_engine.n_evaluators):
                    col = ptu_matrix[:, eval_idx]
                    max_ptu = np.max(col)
                    
                    if max_ptu > threshold:
                        # Find all candidates with this max value
                        tied_candidates = np.where(col == max_ptu)[0].tolist()
                        
                        # Apply tie-breaking with pre-computed holistic scores
                        selected_idx = self._apply_hierarchical_tiebreaker_candidate_centric(
                            tied_candidates,
                            ptu_matrix,
                            eval_idx,
                            score_take,
                            target_query_embedding_similarity,
                            already_selected_cands,
                            holistic_scores_candidate_centric
                        )
                        already_selected_cands.add(selected_idx)
                        
                        src_eval = self.ptu_engine._resolve_source_evaluator_index(
                            self.ptu_engine.candidate_set[selected_idx], selected_idx
                        )
                        winning_source_evals.add(src_eval)
                
                # Collect ALL candidates associated with the winning source samples
                selected_candidate_indices = [
                    idx for idx, cand in enumerate(self.ptu_engine.candidate_set)
                    if self.ptu_engine._resolve_source_evaluator_index(cand, idx) in winning_source_evals
                ]
                subset_size = len(winning_source_evals)
                
            else:  # Evaluator_Centric
                # Find max PTU for each candidate (row maxima)
                winning_evals = set()
                
                evaluator_max_counts = self._compute_evaluator_maximum_counts(ptu_matrix, threshold)
                for cand_idx in range(self.ptu_engine.n_candidates):
                    row = ptu_matrix[cand_idx, :]
                    max_ptu = np.max(row)
                    
                    if max_ptu > threshold:
                        # Find all evaluators with this max value
                        tied_evaluators = np.where(row == max_ptu)[0].tolist()
                        
                        # Apply tie-breaking
                        selected_idx = self._apply_hierarchical_tiebreaker_evaluator_centric(
                            tied_evaluators,
                            ptu_matrix,
                            cand_idx,
                            score_make,
                            target_query_embedding_similarity,
                            evaluator_max_counts
                        )
                        winning_evals.add(selected_idx)
                
                # Collect ALL candidates associated with the optimal subset of evaluators
                selected_candidate_indices = [
                    idx for idx, cand in enumerate(self.ptu_engine.candidate_set)
                    if self.ptu_engine._resolve_source_evaluator_index(cand, idx) in winning_evals
                ]
                subset_size = len(winning_evals)
        
        # Evaluate
        selected_candidate_ids = self.ptu_engine.candidate_indices_to_ids(selected_candidate_indices)
        exemplar_ids_c, exemplar_texts_c = self.ptu_engine.get_source_exemplars(selected_candidate_indices)
        pass_at_n = None
        cand_texts_c = self.ptu_engine.get_candidate_texts(selected_candidate_indices)
        
        # Determine which scores to use based on perspective and fallback condition
        if zero_score_fallback_triggered:
            # In fallback mode, use score_take (all zeros anyway)
            score_take = self.ptu_engine.compute_score_take(ptu_matrix)
            log_scores = score_take
        else:
            # Normal mode: Pick the right scores depending on the strategy
            score_take = self.ptu_engine.compute_score_take(ptu_matrix)
            score_make = self.ptu_engine.compute_score_make(ptu_matrix)
            log_scores = score_take if perspective == 'Candidate_Centric' else score_make
        
        result = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            utility_calibration=utility_calibration,
            evaluator_setting=mask_type,
            scoring_strategy=perspective,
            application=f"Block_C_{perspective}",
            subset_size=subset_size,
            selected_candidates=selected_candidate_ids,
            selected_scores=[float(log_scores[idx]) for idx in selected_candidate_indices],
            selected_exemplar_ids=exemplar_ids_c,
            selected_exemplar_texts=exemplar_texts_c,
            selected_candidate_texts=cand_texts_c,
            zero_score_fallback_triggered=zero_score_fallback_triggered,
            list_ap_score=None,
            group_pass_at_n=pass_at_n
        )
        results.append(result)
        
        logger.info(
            f"Block C - {mask_type} {perspective}: "
            f"K_optimal = {subset_size}, "
            f"Fallback_Triggered = {zero_score_fallback_triggered}, "
            f"Pass@{self.config.global_pass_at_N} = PENDING"
        )
        
        self.results.append(result)
        return results



# THREAD AGGREGATOR (NEW)

class ThreadAggregator:
    """
    Aggregates ExperimentResult objects into per-configuration-thread statistics.
    Prepares data for comprehensive CSV report generation.
    """
    
    @staticmethod
    def aggregate_results(all_results: List[ExperimentResult], config: Layer2Config) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate results by configuration thread.
        
        Args:
            all_results: List of all ExperimentResult objects from orchestration
            config: Layer2Config containing global settings
        
        Returns:
            Dict mapping configuration_thread_name -> aggregated metrics dict
        """
        agg_data = defaultdict(lambda: {
            "calibration": None,
            "mask": None,
            "block": None,
            "strategy": None,
            "top_k": None,
            "threshold": None,
            "total_queries": 0,
            "pass_at_metrics": defaultdict(float),  # {k: sum_of_passes}
            "ap_scores_reranked": [],
            "ap_scores_original": [],
            "ap_improvements": [],
            "coverage_rates": [],
            "position_shifts": []
        })
        
        # Aggregate results by configuration thread
        for result in all_results:
            # Create unique thread identifier
            thread_key = f"{result.utility_calibration}_{result.evaluator_setting}_{result.application}_{result.scoring_strategy}"
            
            # Parse block and application details
            if "Block_A" in result.application:
                block = "Block_A"
                if "Reranking" in result.application:
                    top_k = None
                else:
                    # Look for the number dynamically
                    parts = result.application.split("_")
                    top_k = None
                    for part in parts:
                        if part.isdigit():
                            top_k = int(part)
                            break
            elif "Block_B" in result.application:
                block = "Block_B"
                top_k = None  # Dynamic K
            elif "Block_C" in result.application:
                block = "Block_C"
                top_k = None
            else:
                block = None
                top_k = None
            
            # Store metadata
            agg_data[thread_key]["calibration"] = result.utility_calibration
            agg_data[thread_key]["mask"] = result.evaluator_setting
            agg_data[thread_key]["block"] = block
            agg_data[thread_key]["strategy"] = result.scoring_strategy
            agg_data[thread_key]["top_k"] = top_k
            
            # Aggregate counts
            agg_data[thread_key]["total_queries"] += 1
            
            # Aggregate Pass@K metrics
            for k, val in result.pass_at_k_metrics.items():
                agg_data[thread_key]["pass_at_metrics"][k] += val
            
            # Aggregate Block A specific metrics
            if result.ap_score_reranked is not None:
                agg_data[thread_key]["ap_scores_reranked"].append(result.ap_score_reranked)
            if result.ap_score_original is not None:
                agg_data[thread_key]["ap_scores_original"].append(result.ap_score_original)
            if result.ap_improvement is not None:
                agg_data[thread_key]["ap_improvements"].append(result.ap_improvement)
            
            # Aggregate additional analysis metrics
            if result.candidate_coverage_rate is not None:
                agg_data[thread_key]["coverage_rates"].append(result.candidate_coverage_rate)
            if result.avg_rerank_position_shift is not None:
                agg_data[thread_key]["position_shifts"].append(result.avg_rerank_position_shift)
        
        # Convert aggregated lists to averages
        for thread_key, data in agg_data.items():
            total_q = data["total_queries"]
            
            # Averages
            data["avg_ap_reranked"] = np.mean(data["ap_scores_reranked"]) if data["ap_scores_reranked"] else None
            data["avg_ap_original"] = np.mean(data["ap_scores_original"]) if data["ap_scores_original"] else None
            data["avg_ap_improvement"] = np.mean(data["ap_improvements"]) if data["ap_improvements"] else None
            
            # Average additional metrics
            data["avg_coverage_rate"] = np.mean(data["coverage_rates"]) if data["coverage_rates"] else None
            data["avg_position_shift"] = np.mean(data["position_shifts"]) if data["position_shifts"] else None
            
            # Convert Pass@K sum to average
            for k in data["pass_at_metrics"]:
                data["pass_at_metrics"][k] /= total_q
        
        return dict(agg_data)
    
    @staticmethod
    def sort_threads_hierarchically(threads: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Sort threads hierarchically: Calibration -> Evaluator_Mask -> Experimental_Block -> Strategy_Name
        
        Args:
            threads: Dict of aggregated thread data
        
        Returns:
            List of (thread_name, thread_data) tuples sorted hierarchically
        """
        calib_order = {"Baseline": -1, "Marginal": 0, "Absolute": 1} 
        mask_order = {"Baseline": -1, "Self": 0, "Others": 1, "All": 2} 
        block_order = {"Block_A": 0, "Block_B": 1, "Block_C": 2}
        
        def sort_key(item):
            thread_name, data = item
            calib_idx = calib_order.get(data.get("calibration"), 2)
            mask_idx = mask_order.get(data["mask"], 3)
            block_idx = block_order.get(data["block"], 3)
            strategy = data["strategy"] or ""
            top_k = data["top_k"] or 0
            return (calib_idx, mask_idx, block_idx, strategy, top_k)
        
        return sorted(threads.items(), key=sort_key)



# LAYER 2 ORCHESTRATOR


class Layer2Orchestrator:
    """
    Master orchestrator for Layer 2 experiments.
    Manages the grid search across all base conditions and blocks.
    """
    
    def __init__(self, config: Layer2Config, output_dir: str, api_manager_solve: Any, api_manager_eval: Any, global_config: Optional[Dict[str, Any]] = None, exemplar_data: Optional[Dict[str, Any]] = None, hard_questions: Optional[List[str]] = None):
        self.config = config
        self.output_dir = output_dir
        self.api_manager_solve = api_manager_solve
        self.api_manager_eval = api_manager_eval
        self.global_config = global_config if global_config is not None else GLOBAL_CONFIG
        self.exemplar_data = exemplar_data or {}       # <--- ADDED
        self.hard_questions = hard_questions or []     # <--- ADDED
        self.inference_engine = ActiveInferenceEngine(api_manager_solve, api_manager_eval, self.global_config)
        self.all_results = []
        os.makedirs(output_dir, exist_ok=True)
    
    def _map_dynamic_method_to_block_a_strategy(self, dynamic_method: str) -> Optional[str]:
        """Map Block B dynamic-K methods to the corresponding Block A ranking strategy."""
        mapping = {
            'K_take': 'ScoreTake',
            'K_make': 'ScoreMake',
            'K_both': 'Holistic'
        }
        return mapping.get(dynamic_method)
    
    def run_single_query(self, layer1_state: Dict[str, Any]) -> List[ExperimentResult]:
        """
        Run all configured experiments for a single query.
        """
        query_results = []
        
        # Strict execution: Let errors crash the pipeline so we catch Layer 1 API failures immediately.
        ptu_engine = PTUMathEngine(layer1_state, self.exemplar_data, self.hard_questions) 
        
        # === NEW: RUN ORIGINAL BASELINE EXACTLY ONCE PER QUESTION ===
        if self.config.run_block_A and getattr(self.config, 'run_block_A_baseline', False):
            logger.info(f"\n[BLOCK A BASELINE] Executing Original Retrieval Baseline")
            block_a_baseline = BlockA(ptu_engine, self.config)
            baseline_results = block_a_baseline.run_original_baseline()
            query_results.extend(baseline_results)
        # ============================================================

        # Grid search: For each calibration mode, then for each mask type
        for calib_mode in self.config.utility_calibration_modes:
            # 1. Fetch mathematically calibrated base PTU
            calibrated_ptu = ptu_engine.get_calibrated_ptu_matrix(calib_mode)
            
            for mask_type in self.config.evaluator_masking:
                # 2. Apply Masking to the calibrated PTU
                masked_ptu = ptu_engine.apply_evaluator_mask(mask_type, calibrated_ptu)
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing Query #{ptu_engine.target_query_idx}: {calib_mode} | {mask_type} Evaluation")
                logger.info(f"{'='*70}")
                
                # ===== BLOCK A =====
                if self.config.run_block_A:
                    logger.info("\n[BLOCK A] Baseline Reranking & Static Grouping")
                    block_a = BlockA(ptu_engine, self.config)
                    
                    for strategy in self.config.block_A_strategies:
                        results = block_a.run_for_mask_and_strategy(
                            masked_ptu, calib_mode, mask_type, strategy,
                            self.config.block_A_weight_taker,
                            self.config.block_A_weight_maker
                        )
                        query_results.extend(results)
                else:
                    block_a = None
                
                # BLOCK B 
                if self.config.run_block_B:
                    logger.info("\n[BLOCK B] Dynamic Smart-K Grouping")
                    block_b = BlockB(ptu_engine, self.config)
                    
                    # Get ranked list for boundary test from Block A using the matching strategy for each method.
                    ranked_list_cache = {}
                    if self.config.run_boundary_intersection_test and block_a:
                        for method in self.config.dynamic_k_methods:
                            matching_strategy = self._map_dynamic_method_to_block_a_strategy(method)
                            if matching_strategy:
                                ranked_indices = block_a.get_ranked_indices_for_strategy(mask_type, matching_strategy)
                                if ranked_indices is None:
                                    logger.warning(
                                        f"Could not retrieve cached ranked indices for Block A strategy '{matching_strategy}' "
                                        f"needed by Block B boundary test for method '{method}'. "
                                        f"Ensure Block A is enabled and includes that strategy."
                                    )
                                ranked_list_cache[method] = ranked_indices
                            else:
                                ranked_list_cache[method] = None
                    
                    for method in self.config.dynamic_k_methods:
                        ranked_list = ranked_list_cache.get(method) if self.config.run_boundary_intersection_test else None
                        results = block_b.run_for_mask_and_method(
                            masked_ptu, calib_mode, mask_type, method,
                            ranked_list,
                            self.config.block_B_weight_taker,
                            self.config.block_B_weight_maker
                        )
                        query_results.extend(results)
                
                # BLOCK C 
                if self.config.run_block_C:
                    logger.info("\n[BLOCK C] Optimal Subset (Coverage) Grouping")
                    block_c = BlockC(ptu_engine, self.config)
                    
                    target_query_embedding_sim = ptu_engine.get_target_query_embedding_similarity()
                    
                    for perspective in self.config.coverage_perspectives:
                        results = block_c.run_for_mask_and_perspective(
                            masked_ptu, calib_mode, mask_type, perspective, target_query_embedding_sim
                        )
                        query_results.extend(results)
        
        # ===== ACTIVE INFERENCE EXECUTION =====
        logger.info(f"\n[ACTIVE INFERENCE] Executing LLM for {len(query_results)} configurations...")
        for result in query_results:
            logger.info(f"  -> Running LLM for: {result.application} (Pass@{self.config.global_pass_at_N})")
            
            # THE PROXY PRINCIPLE FIX: Pass the Source Exemplars to the Solver, NOT the drafted Candidates!
            prompt, executions, pass_at_k = self.inference_engine.execute_and_evaluate(
                target_query=result.target_query_text,
                ground_truth=result.ground_truth_answer,
                context_texts=result.selected_exemplar_texts, 
                n_attempts=self.config.global_pass_at_N
            )
            
            # Attach live payloads to the result object
            result.final_prompt_text = prompt
            result.executions = executions
            result.pass_at_k_metrics = pass_at_k
            
            # For backward compatibility with the legacy summary function
            result.group_pass_at_n = pass_at_k.get(self.config.global_pass_at_N, 0.0)

        self.all_results.extend(query_results)
        return query_results
    
    def generate_master_report(self) -> Dict[str, Any]:
        """
        Generate detailed JSON report matching the strict schema requirements.
        """
        results_data = []
        for r in self.all_results:
            # Map strictly to the required schema
            record = {
                "main_question_id": r.target_query_idx,
                "configuration_thread": f"{r.utility_calibration}_{r.evaluator_setting}_{r.application}_{r.scoring_strategy}",
                "utility_calibration_mode": r.utility_calibration,
                "context_selection_metadata": {
                    "selected_candidate_proxies": r.selected_candidates,
                    "selected_exemplar_ids": r.selected_exemplar_ids,
                    "selected_scores": r.selected_scores,
                    "subset_size": r.subset_size,
                    "zero_score_fallback_triggered": r.zero_score_fallback_triggered
                },
                "context_payload_texts": r.selected_exemplar_texts,
                "final_prompt_text": r.final_prompt_text,
                "executions": r.executions
            }
            results_data.append(self._make_serializable(record))
        
        report = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_experiments": len(self.all_results),
                "total_queries": len(set(r.target_query_idx for r in self.all_results)),
                "config": self._make_serializable(asdict(self.config))
            },
            "experiments": results_data
        }
        
        return report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with AP metrics, ranking metrics, and all aggregations.
        Returns data structure ready for CSV conversion.
        
        Returns:
            Dict mapping thread_name -> comprehensive metrics
        """
        # Aggregate results by configuration thread
        aggregated = ThreadAggregator.aggregate_results(self.all_results, self.config)
        
        # Sort threads hierarchically
        sorted_threads = ThreadAggregator.sort_threads_hierarchically(aggregated)
        
        comprehensive_report = {}
        for thread_name, thread_data in sorted_threads:
            comprehensive_report[thread_name] = thread_data
        
        return comprehensive_report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy and other non-serializable types for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def save_reports(self, report: Dict[str, Any], json_filename: str = None, csv_filename: str = None):
        """Save the detailed JSON logs and generate both the existing and comprehensive CSV reports safely."""
        
        json_filename = f"layer2_detailed_logs_{self.config.layer2_config_name}.json"
        csv_filename = f"layer2_master_report_{self.config.layer2_config_name}.csv"

        # 1. Save JSON (Atomically)
        json_filepath = os.path.join(self.output_dir, json_filename)
        temp_json_filepath = json_filepath + ".tmp"
        save_json(report, temp_json_filepath)
        os.replace(temp_json_filepath, json_filepath)
        logger.debug(f"Detailed JSON logs updated: {json_filepath}")

        # 2. Generate and Save EXISTING CSV (Atomically)
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        temp_csv_filepath = csv_filepath + ".tmp"
        
        agg_data = defaultdict(lambda: {
            "Total_Questions": 0, 
            "Pass_At_Metrics": defaultdict(float)
        })
        
        max_k = 0
        for r in self.all_results:
            thread_name = f"{r.application}_{r.scoring_strategy}"
            agg_data[thread_name]["Total_Questions"] += 1
            for k, val in r.pass_at_k_metrics.items():
                agg_data[thread_name]["Pass_At_Metrics"][k] += val
                if k > max_k: max_k = k

        with open(temp_csv_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            expected_n = self.config.global_pass_at_N
            headers = ["Configuration_Thread", "Total_Questions_Evaluated"]
            for k in range(1, expected_n + 1):
                headers.append(f"Pass@{k}_Accuracy")
            writer.writerow(headers)
            
            for thread_name, stats in agg_data.items():
                total_q = stats["Total_Questions"]
                row = [thread_name, total_q]
                for k in range(1, expected_n + 1):
                    avg_pass_k = stats["Pass_At_Metrics"][k] / total_q if total_q > 0 else 0
                    row.append(round(avg_pass_k, 4))
                writer.writerow(row)
                
        # Instantly overwrite the old file with the new one
        os.replace(temp_csv_filepath, csv_filepath)

        # 3. Generate and Save NEW COMPREHENSIVE CSV
        self._save_comprehensive_report()
        
        return json_filepath, csv_filepath
    
    def _save_comprehensive_report(self, csv_filename: str = None):
        """
        Generate and save the comprehensive analysis CSV with AP metrics, ranking metrics, and hierarchical organization.
        """
        # --- ADD THIS LINE TO MAKE THE NAME DYNAMIC ---
        csv_filename = f"layer2_comprehensive_analysis_{self.config.layer2_config_name}.csv"
        # ----------------------------------------------

        comprehensive_data = self.generate_comprehensive_report()
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        
        # Prepare headers
        expected_n = self.config.global_pass_at_N
        
        headers = [
            # Group 1: Configuration Identity
            "Utility_Calibration", "Evaluator_Mask", "Experimental_Block", "Strategy_Name",
            # Group 2: Block-Specific Hyperparameters
            "Top_K", "Threshold",
            # Group 3: Global Execution Settings
            "Global_Pass_at_N",
            # Group 4: Execution Scope
            "Total_Queries_Evaluated",
        ]
        
        # Group 6: Pass@N Metrics
        for k in range(1, expected_n + 1):
            headers.append(f"Pass_At_{k}")
        
        # Group 7: AP Analysis (Block A only)
        headers.extend(["AP_Score_Reranked", "AP_Score_Original", "AP_Improvement"])
        
        # Group 9: Additional Analysis
        headers.extend(["Candidate_Coverage_Rate", "Avg_Rerank_Position_Shift"])
        
        # Write to CSV (Atomically)
        temp_csv_filepath = csv_filepath + ".tmp"
        with open(temp_csv_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            # Data rows (already sorted hierarchically)
            for thread_name, thread_data in comprehensive_data.items():
                row = [
                    # Group 1: Configuration Identity
                    thread_data.get("calibration", ""),
                    thread_data.get("mask", ""),
                    thread_data.get("block", ""),
                    thread_data.get("strategy", ""),
                    # Group 2: Block-Specific Hyperparameters
                    thread_data.get("top_k") if thread_data.get("top_k") is not None else "-",
                    "-",  # Threshold (not used in current config; set to N/A)
                    # Group 3: Global Execution Settings
                    self.config.global_pass_at_N,
                    # Group 4: Execution Scope
                    thread_data.get("total_queries", 0),
                ]
                
                # Group 6: Pass@N Metrics
                for k in range(1, expected_n + 1):
                    pass_k_val = thread_data.get("pass_at_metrics", {}).get(k, 0.0)
                    row.append(round(pass_k_val, 4))
                
                # Group 7: AP Analysis (Block A only; "-" for others)
                ap_reranked = thread_data.get("avg_ap_reranked")
                ap_original = thread_data.get("avg_ap_original")
                ap_improvement = thread_data.get("avg_ap_improvement")
                row.extend([
                    round(ap_reranked, 4) if ap_reranked is not None else "-",
                    round(ap_original, 4) if ap_original is not None else "-",
                    round(ap_improvement, 4) if ap_improvement is not None else "-",
                ])
                
                # Group 9: Additional Analysis
                coverage = thread_data.get("avg_coverage_rate")
                position_shift = thread_data.get("avg_position_shift")
                row.extend([
                    round(coverage, 4) if coverage is not None else "-",
                    round(position_shift, 4) if position_shift is not None else "-",
                ])
                
                writer.writerow(row)
        
        # Instantly overwrite the old file with the new one
        os.replace(temp_csv_filepath, csv_filepath)
        logger.debug(f"Comprehensive analysis CSV report updated: {csv_filepath}")



    def get_checkpoint_path(self) -> str:
            """Returns the file path for the internal memory checkpoint."""
            return os.path.join(self.output_dir, f"layer2_internal_checkpoint_{self.config.layer2_config_name}.pkl")

    def save_checkpoint(self):
        """Saves the current memory state to a file atomically."""
        path = self.get_checkpoint_path()
        temp_path = path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(self.all_results, f)
        os.replace(temp_path, path)

    def load_checkpoint(self) -> bool:
        """Loads memory state from a previous run if the kernel crashed."""
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    self.all_results = pickle.load(f)
                print(f"🔄 Checkpoint loaded! Recovered {len(self.all_results)} experiment configs from previous run.")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")
        return False

# ============================================================================
# PUBLIC API
# ============================================================================

def run_layer2_experiments(
    layer1_states: List[Dict[str, Any]],
    config: Layer2Config,
    output_dir: str,
    api_manager_solve: Any,
    api_manager_eval: Any,
    global_config: Optional[Dict[str, Any]] = None,
    exemplar_data: Optional[Dict[str, Any]] = None,
    hard_questions: Optional[List[str]] = None       
) -> Tuple[List[ExperimentResult], Dict[str, Any]]:
    """
    Run complete Layer 2 analysis on Layer 1 cached states with Iterative Saving.
    """
    run_config = global_config if global_config is not None else GLOBAL_CONFIG
    orchestrator = Layer2Orchestrator(config, output_dir, api_manager_solve, api_manager_eval, run_config, exemplar_data, hard_questions) 
    
    # 1. Attempt to load previous progress if kernel crashed
    orchestrator.load_checkpoint()
    
    # 2. Filter out questions we have already completely finished
    completed_query_ids = {res.target_query_idx for res in orchestrator.all_results}
    pending_states = [state for state in layer1_states if state.get('target_query_idx') not in completed_query_ids]
    
    total_queries = len(layer1_states)
    pending_count = len(pending_states)
    
    print("\n" + "="*60)
    print(f"🚀 STARTING LAYER 2 ANALYSIS")
    print(f"Total questions: {total_queries} | Already done: {total_queries - pending_count} | Remaining: {pending_count}")
    print("="*60)
    
    if pending_count > 0:
        for loop_idx, layer1_state in enumerate(tqdm(pending_states, desc="Layer 2 Progress")):
            
            # Run the actual question
            orchestrator.run_single_query(layer1_state)
            
            # --- ITERATIVE SAVING (Happens after EVERY question) ---
            # 1. Save memory checkpoint (.pkl)
            orchestrator.save_checkpoint()
            
            # 2. Overwrite JSON and CSV files on disk so you can watch them update live
            report = orchestrator.generate_master_report()
            orchestrator.save_reports(report)
            
            # Trigger HuggingFace Sync if it's time!
            periodic_sync_check(loop_idx, run_config)
    
    print("\n" + "="*60)
    print("✅ LAYER 2 ANALYSIS COMPLETELY FINISHED!")
    print("="*60)
    
    # Final generation just to be safe and return the final report variable
    report = orchestrator.generate_master_report()
    orchestrator.save_reports(report)
    
    # Final sync at the very end to make sure the final CSVs get uploaded
    if run_config.get("PERSIST_RESULTS_ONLINE"):
        print("\n--- Final Sync: Forcing workspace backup to Hugging Face Hub ---")
        from src.hf_sync import sync_workspace_to_hub
        sync_workspace_to_hub(run_config)
    
    return orchestrator.all_results, report