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
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import concurrent.futures

from src.utils import save_json, load_json, convert_numpy_for_json
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.evaluation import evaluate_single_answer_with_llm
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
    evaluator_setting: str  # 'Self', 'Others', 'All'
    scoring_strategy: str   # 'ScoreTake', 'ScoreMake', 'Holistic'
    application: str = ""  # 'Block_A_Reranking', 'Block_A_TopK', 'Block_B_Dynamic', etc.
    weight_taker: float = 1.0
    weight_maker: float = 1.0
    subset_size: int = 0
    selected_candidates: List[str] = None  # Dataset IDs of selected candidates
    selected_evaluators: List[int] = None  # Indices of selected evaluators
    
    # Evaluation Metrics (Legacy)
    list_ap_score: Optional[float] = None  
    group_pass_at_n: Optional[float] = None  
    
# --- ACTIVE INFERENCE PAYLOADS (NEW) ---
    selected_candidate_texts: List[str] = None
    final_prompt_text: Optional[str] = None
    zero_score_fallback_triggered: bool = False
    executions: List[Dict[str, Any]] = None  
    pass_at_k_metrics: Dict[int, float] = None
    
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
        if self.selected_candidate_texts is None:
            self.selected_candidate_texts = []
        if self.executions is None:
            self.executions = []
        if self.pass_at_k_metrics is None:
            self.pass_at_k_metrics = {}


@dataclass
class Layer2Config:
    """Configuration for Layer 2 experiments."""
    # Block Execution Toggles
    run_block_A: bool = True
    run_block_B: bool = True
    run_block_C: bool = True
    
    # Global Base Conditions
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
# PTU MATH ENGINE
# ============================================================================

class PTUMathEngine:
    """
    Computes Pairwise Transaction Utility (PTU) matrices and score aggregations
    from Layer 1 cached data.
    """
    
    def __init__(self, layer1_state: Dict[str, Any]):
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
        self.target_query_idx = layer1_state.get('target_query_idx')
        self.target_query_text = layer1_state.get('target_query_text')
        self.ground_truth_answer = layer1_state.get('ground_truth_answer')
        
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
        
        # Compute core PTU matrix
        self.ptu_matrix = self._compute_ptu_matrix()
        
        # Cache for computed scores
        self._score_cache = {}
        
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
            try:
                items.sort(key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            except Exception:
                pass

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
            try:
                items.sort(key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            except Exception:
                pass

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
        """Fetch a cross-evaluation score from Layer 1 state in a robust way."""
        row = self._lookup_value_by_key_variants(self.cross_eval_matrix, candidate_id, {})
        if isinstance(row, dict):
            score = self._lookup_value_by_key_variants(row, evaluator_id, 0.0)
            return float(score) if score is not None else 0.0
        # Support tuple-keyed cross evaluation matrices
        tuple_key = (candidate_id, evaluator_id)
        score = self._lookup_value_by_key_variants(self.cross_eval_matrix, tuple_key, None)
        if score is not None:
            return float(score)
        score = self._lookup_value_by_key_variants(self.cross_eval_matrix, str(tuple_key), 0.0)
        return float(score) if score is not None else 0.0
    
    def _fetch_intrinsic_baseline(self, evaluator_id: Any) -> float:
        """Fetch intrinsic baseline score from Layer 1 state."""
        score = self._lookup_value_by_key_variants(self.intrinsic_baselines, evaluator_id, 0.0)
        return float(score) if score is not None else 0.0
    
    def _compute_ptu_matrix(self) -> np.ndarray:
        """
        Compute the base PTU matrix without masking.
        
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
                
                # Validate fetched scores are numeric
                if not isinstance(induced_ccs, (int, float)) or np.isnan(induced_ccs):
                    logger.warning(
                        f"PTU computation: Invalid induced_ccs for candidate {cand_id} "
                        f"and evaluator {eval_id}. Using 0.0."
                    )
                    induced_ccs = 0.0
                
                if not isinstance(intrinsic_ccs, (int, float)) or np.isnan(intrinsic_ccs):
                    logger.warning(
                        f"PTU computation: Invalid intrinsic_ccs for evaluator {eval_id}. "
                        f"Using 0.0."
                    )
                    intrinsic_ccs = 0.0
                
                ptu[cand_idx, eval_idx] = max(0.0, induced_ccs - intrinsic_ccs)
        
        # Final validation: Ensure matrix is fully populated
        if np.all(ptu == 0.0):
            logger.warning(
                "PTU matrix computation completed, but all values are zero. "
                "This may indicate empty or mismatched cross-evaluation data in Layer 1 cache."
            )
        
        return ptu
    
    def apply_evaluator_mask(self, mask_type: str) -> np.ndarray:
        """
        Apply evaluator masking to the PTU matrix.
        
        Args:
            mask_type: 'Self' (diagonal only), 'Others' (off-diagonal), or 'All' (no mask)
        
        Returns:
            Masked PTU matrix
        
        Raises:
            ValueError: If mask_type is invalid
            RuntimeError: If matrix dimensions are inconsistent
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
        if self.ptu_matrix is None:
            raise RuntimeError(
                "apply_evaluator_mask failed: PTU matrix is None. "
                "The math engine must be properly initialized before applying masks."
            )
        
        if self.ptu_matrix.shape != (self.n_candidates, self.n_evaluators):
            raise RuntimeError(
                f"apply_evaluator_mask failed: PTU matrix shape mismatch. "
                f"Expected ({self.n_candidates}, {self.n_evaluators}) "
                f"but got {self.ptu_matrix.shape}."
            )
        
        # Create a copy to avoid modifying the original
        masked_ptu = self.ptu_matrix.copy()
        
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
        """Return the cached target query similarity score for each retrieval/evaluator."""
        similarity_map: Dict[int, float] = {}
        for evaluator_idx, evaluator in enumerate(self.retrieved_set):
            similarity_score = 0.0
            if isinstance(evaluator, dict):
                similarity_score = float(evaluator.get('similarity_score', 0.0) or 0.0)
            similarity_map[evaluator_idx] = similarity_score
        return similarity_map

    def _resolve_source_evaluator_index(self, candidate: Dict[str, Any], default_idx: int) -> int:
        """Resolve the matrix evaluator index for a candidate's source exemplar."""
        source_id = candidate.get('source_exemplar_idx') if isinstance(candidate, dict) else None
        if source_id is None:
            return default_idx
        source_key = self._normalize_id(source_id)
        return self.evaluator_id_to_idx.get(source_key, default_idx)
    
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


def _normalize_ground_truth_label(label_obj: Any) -> bool:
    """Normalize a ground truth label value into a boolean is_correct flag."""
    if isinstance(label_obj, dict):
        return bool(label_obj.get('is_correct', False))
    return bool(label_obj)


def _get_ground_truth_label(ground_truth_labels: Dict[Any, Any], cand_idx: int) -> bool:
    """Fetch and normalize a candidate's ground truth correctness label."""
    if cand_idx in ground_truth_labels:
        return _normalize_ground_truth_label(ground_truth_labels[cand_idx])
    cand_key = str(cand_idx)
    if cand_key in ground_truth_labels:
        return _normalize_ground_truth_label(ground_truth_labels[cand_key])
    return False


# ============================================================================
# ACTIVE INFERENCE ENGINE (LIVE LLM EVALUATION)
# ============================================================================

class ActiveInferenceEngine:
    """
    Executes live API calls to the LLM using the contexts chosen by Layer 2 Blocks.
    """
    def __init__(self, api_manager_solve: Any, api_manager_eval: Any):
        self.api_manager_solve = api_manager_solve
        self.api_manager_eval = api_manager_eval
        self.global_config = GLOBAL_CONFIG
        
        # Determine which model to use based on the API Manager type
        if isinstance(self.api_manager_solve, GeminiAPIManager):
            self.model_name = self.global_config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
        elif isinstance(self.api_manager_solve, AvalAIAPIManager):
            self.model_name = self.global_config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
        else:
            self.model_name = self.global_config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')

def execute_and_evaluate(self, target_query: str, ground_truth: str, context_texts: List[str], n_attempts: int):
        # 1. Prompt Assembly
        if not context_texts:
            # Naked prompt (Zero-Score Fallback or no context)
            prompt = create_final_reasoning_prompt_simple(target_query, self.global_config)
        else:
            # Format the raw candidate answers so the LLM understands what they are
            formatted_contexts = []
            for text in context_texts:
                # If it's already formatted, leave it alone, otherwise wrap it
                if "Question:" in text and "Rationale and Answer:" in text:
                    formatted_contexts.append(text)
                else:
                    formatted_contexts.append(f"Question: {target_query}\nRationale and Answer: {text}")
                    
            # Context-Aware prompt
            prompt = create_final_reasoning_prompt(target_query, formatted_contexts, self.global_config)

        executions = []
        temp = self.global_config.get("DEFAULT_PASS_N_SOLVER_TEMPERATURE", 1.0)

        # 2. Worker function for concurrent execution
        def run_attempt(i):
            resp = self.api_manager_solve.generate_content(prompt, self.model_name, temp)
            is_correct = False
            raw_text = resp.get('text', '')
            error_msg = resp.get('error_message', '')
            
            if resp['status'] == 'SUCCESS':
                eval_res = evaluate_single_answer_with_llm(raw_text, ground_truth, self.api_manager_eval, self.global_config)
                is_correct = eval_res.get('is_correct', False)
                
            return {
                "attempt_index": i + 1,
                "raw_llm_generation": raw_text,
                "error": error_msg,
                "is_correct": is_correct
            }

        # 3. Execute N Independent Inferences Concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_attempts, 10)) as executor:
            futures = [executor.submit(run_attempt, i) for i in range(n_attempts)]
            for future in concurrent.futures.as_completed(futures):
                executions.append(future.result())
        
        # Sort executions sequentially
        executions.sort(key=lambda x: x["attempt_index"])

        # 4. Calculate empirical Pass@k metrics
        pass_at_k = {}
        is_correct_array = [ex["is_correct"] for ex in executions]
        for k in range(1, n_attempts + 1):
            pass_at_k[k] = 1.0 if any(is_correct_array[:k]) else 0.0

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
    
    def run_for_mask_and_strategy(
        self,
        ptu_matrix: np.ndarray,
        mask_type: str,
        strategy: str,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> List[ExperimentResult]:
        """
        Run Block A experiments for a specific mask and strategy combination.
        """
        results = []
        
        # Get scores for this strategy
        scores = self.ptu_engine.get_scores_for_strategy(
            ptu_matrix, strategy, weight_taker, weight_maker
        )
        
        # Experiment A.1: Reranking & Average Precision
        ranked_indices = np.argsort(-scores)  # Sort descending
        
        # Cache the ranked indices for Block B boundary test (Experiment B.2.3)
        cache_key = (mask_type, strategy)
        self.ranked_indices_cache[cache_key] = ranked_indices.tolist()
        
        ranked_candidate_ids = self.ptu_engine.candidate_indices_to_ids(ranked_indices.tolist())
        ap_score = None
        
        cand_texts_a1 = self.ptu_engine.get_candidate_texts(ranked_indices.tolist())
        result_a1 = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            evaluator_setting=mask_type,
            scoring_strategy=strategy,
            weight_taker=weight_taker,
            weight_maker=weight_maker,
            application=f"Block_A_Reranking_{strategy}",
            subset_size=len(ranked_indices),
            selected_candidates=ranked_candidate_ids,
            selected_candidate_texts=cand_texts_a1,
            zero_score_fallback_triggered=(len(cand_texts_a1) == 0),
            list_ap_score=ap_score,
            group_pass_at_n=None
        )
        results.append(result_a1)
        
        logger.info(
            f"Block A A.1 - {mask_type} {strategy}: "
            f"AP = {ap_score:.4f}"
        )
        
        # Experiment A.2: Static Top-K Grouping
        for k in self.config.top_ks_group:
            if k > len(ranked_indices):
                continue
            
            top_k_indices = ranked_indices[:k].tolist()
            top_k_candidate_ids = self.ptu_engine.candidate_indices_to_ids(top_k_indices)
            pass_at_n = None
            
            cand_texts_a2 = self.ptu_engine.get_candidate_texts(top_k_indices)
            result_a2 = ExperimentResult(
                target_query_idx=self.ptu_engine.target_query_idx,
                target_query_text=self.ptu_engine.target_query_text,
                ground_truth_answer=self.ptu_engine.ground_truth_answer,
                evaluator_setting=mask_type,
                scoring_strategy=strategy,
                weight_taker=weight_taker,
                weight_maker=weight_maker,
                application=f"Block_A_TopK_{k}_{strategy}",
                subset_size=k,
                selected_candidates=top_k_candidate_ids,
                selected_candidate_texts=cand_texts_a2,
                zero_score_fallback_triggered=(len(cand_texts_a2) == 0),
                list_ap_score=None,
                group_pass_at_n=pass_at_n
            )
            results.append(result_a2)
            
            logger.info(
                f"Block A A.2 - {mask_type} {strategy} Top-{k}: "
                f"Pass@{self.config.global_pass_at_N} = {pass_at_n:.4f}"
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


# ============================================================================
# BLOCK B: DYNAMIC SMART-K GROUPING
# ============================================================================

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
        mask_type: str,
        method: str,
        ranked_list_for_boundary: Optional[List[int]] = None,
        weight_taker: float = 1.0,
        weight_maker: float = 1.0
    ) -> List[ExperimentResult]:
        """
        Run Block B experiments for a specific mask and dynamic-K method.
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
        
        k_dynamic = len(positive_indices)
        zero_score_fallback_triggered = (k_dynamic == 0)
        
        # Evaluate the dynamic group
        selected_candidate_ids = self.ptu_engine.candidate_indices_to_ids(positive_indices)
        pass_at_n = None
        result = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            evaluator_setting=mask_type,
            scoring_strategy=method,
            weight_taker=weight_taker,
            weight_maker=weight_maker,
            application=f"Block_B_Dynamic_{method}",
            subset_size=k_dynamic,
            selected_candidates=selected_candidate_ids,
            selected_candidate_texts=self.ptu_engine.get_candidate_texts(positive_indices),
            zero_score_fallback_triggered=zero_score_fallback_triggered,
            list_ap_score=None,
            group_pass_at_n=pass_at_n
        )
        results.append(result)
        
        logger.info(
            f"Block B - {mask_type} {method}: "
            f"K_dynamic = {k_dynamic}, Pass@{self.config.global_pass_at_N} = {pass_at_n:.4f}"
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


# ============================================================================
# BLOCK C: OPTIMAL SUBSET (COVERAGE) GROUPING
# ============================================================================

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
        mask_type: str,
        perspective: str,
        target_query_embedding_similarity: Dict[int, float]
    ) -> List[ExperimentResult]:
        """
        Run Block C experiments for a specific mask and perspective.
        """
        results = []
        threshold = self.config.activation_threshold
        
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
        pass_at_n = None
        cand_texts_c = self.ptu_engine.get_candidate_texts(selected_candidate_indices)
        result = ExperimentResult(
            target_query_idx=self.ptu_engine.target_query_idx,
            target_query_text=self.ptu_engine.target_query_text,
            ground_truth_answer=self.ptu_engine.ground_truth_answer,
            evaluator_setting=mask_type,
            scoring_strategy=perspective,
            application=f"Block_C_{perspective}",
            subset_size=subset_size,
            selected_candidates=selected_candidate_ids,
            selected_candidate_texts=cand_texts_c,
            zero_score_fallback_triggered=(len(cand_texts_c) == 0),
            list_ap_score=None,
            group_pass_at_n=pass_at_n
        )
        results.append(result)
        
        logger.info(
            f"Block C - {mask_type} {perspective}: "
            f"K_optimal = {subset_size}, Pass@{self.config.global_pass_at_N} = {pass_at_n:.4f}"
        )
        
        self.results.append(result)
        return results


# ============================================================================
# LAYER 2 ORCHESTRATOR
# ============================================================================

class Layer2Orchestrator:
    """
    Master orchestrator for Layer 2 experiments.
    Manages the grid search across all base conditions and blocks.
    """
    
    def __init__(self, config: Layer2Config, output_dir: str, api_manager_solve: Any, api_manager_eval: Any):
        self.config = config
        self.output_dir = output_dir
        self.api_manager_solve = api_manager_solve
        self.api_manager_eval = api_manager_eval
        self.inference_engine = ActiveInferenceEngine(api_manager_solve, api_manager_eval)
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
        
        try:
            ptu_engine = PTUMathEngine(layer1_state)
        except (ValueError, RuntimeError) as e:
            target_idx = layer1_state.get('target_query_idx', 'Unknown')
            logger.error(f"Skipping Layer 2 Analysis for Query #{target_idx}: {e}")
            print(f"  [!] Skipping Query #{target_idx} due to empty/invalid Layer 1 data.")
            return [] # Skip this query and move to the next one
        
        # Grid search: For each mask type
        for mask_type in self.config.evaluator_masking:
            # Apply mask to PTU matrix
            masked_ptu = ptu_engine.apply_evaluator_mask(mask_type)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing Query #{ptu_engine.target_query_idx}: {mask_type} Evaluation")
            logger.info(f"{'='*70}")
            
            # ===== BLOCK A =====
            if self.config.run_block_A:
                logger.info("\n[BLOCK A] Baseline Reranking & Static Grouping")
                block_a = BlockA(ptu_engine, self.config)
                
                for strategy in self.config.block_A_strategies:
                    results = block_a.run_for_mask_and_strategy(
                        masked_ptu, mask_type, strategy,
                        self.config.block_A_weight_taker,
                        self.config.block_A_weight_maker
                    )
                    query_results.extend(results)
            else:
                block_a = None
            
            # ===== BLOCK B =====
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
                        masked_ptu, mask_type, method,
                        ranked_list,
                        self.config.block_B_weight_taker,
                        self.config.block_B_weight_maker
                    )
                    query_results.extend(results)
            
            # ===== BLOCK C =====
            if self.config.run_block_C:
                logger.info("\n[BLOCK C] Optimal Subset (Coverage) Grouping")
                block_c = BlockC(ptu_engine, self.config)
                
                target_query_embedding_sim = ptu_engine.get_target_query_embedding_similarity()
                
                for perspective in self.config.coverage_perspectives:
                    results = block_c.run_for_mask_and_perspective(
                        masked_ptu, mask_type, perspective, target_query_embedding_sim
                    )
                    query_results.extend(results)
        
        # ===== ACTIVE INFERENCE EXECUTION =====
        logger.info(f"\n[ACTIVE INFERENCE] Executing LLM for {len(query_results)} configurations...")
        for result in query_results:
            logger.info(f"  -> Running LLM for: {result.application} (Pass@{self.config.global_pass_at_N})")
            
            prompt, executions, pass_at_k = self.inference_engine.execute_and_evaluate(
                target_query=result.target_query_text,
                ground_truth=result.ground_truth_answer,
                context_texts=result.selected_candidate_texts,
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
                "configuration_thread": f"{r.application}_{r.scoring_strategy}",
                "context_selection_metadata": {
                    "selected_samples": r.selected_candidates,
                    "subset_size": r.subset_size,
                    "zero_score_fallback_triggered": r.zero_score_fallback_triggered
                },
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
    
    def save_reports(self, report: Dict[str, Any], json_filename: str = "layer2_detailed_logs.json", csv_filename: str = "layer2_master_report.csv"):
        """Save both the detailed JSON logs and the aggregated CSV report to disk."""
        # 1. Save JSON
        json_filepath = os.path.join(self.output_dir, json_filename)
        save_json(report, json_filepath)
        logger.info(f"Detailed JSON logs saved to: {json_filepath}")

        # 2. Generate and Save CSV
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        
        # Aggregate data by Configuration Thread
        agg_data = defaultdict(lambda: {
            "Total_Questions": 0, 
            "Total_Context_Size": 0, 
            "Pass_At_Metrics": defaultdict(float)
        })
        
        max_k = 0
        for r in self.all_results:
            thread_name = f"{r.application}_{r.scoring_strategy}"
            agg_data[thread_name]["Total_Questions"] += 1
            agg_data[thread_name]["Total_Context_Size"] += r.subset_size
            
            for k, val in r.pass_at_k_metrics.items():
                agg_data[thread_name]["Pass_At_Metrics"][k] += val
                if k > max_k: max_k = k

        # Write to CSV
        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Create Headers
            headers = ["Configuration_Thread", "Total_Questions_Evaluated", "Average_Context_Samples"]
            for k in range(1, max_k + 1):
                headers.append(f"Pass@{k}_Accuracy")
            writer.writerow(headers)
            
            # Write Rows
            for thread_name, stats in agg_data.items():
                total_q = stats["Total_Questions"]
                avg_context = stats["Total_Context_Size"] / total_q if total_q > 0 else 0
                
                row = [thread_name, total_q, round(avg_context, 2)]
                for k in range(1, max_k + 1):
                    avg_pass_k = stats["Pass_At_Metrics"][k] / total_q if total_q > 0 else 0
                    row.append(round(avg_pass_k, 4))
                    
                writer.writerow(row)

        logger.info(f"Master CSV report saved to: {csv_filepath}")
        return json_filepath, csv_filepath


# ============================================================================
# PUBLIC API
# ============================================================================

def run_layer2_experiments(
    layer1_states: List[Dict[str, Any]],
    config: Layer2Config,
    output_dir: str,
    api_manager_solve: Any,
    api_manager_eval: Any
) -> Tuple[List[ExperimentResult], Dict[str, Any]]:
    """
    Run complete Layer 2 analysis on Layer 1 cached states.
    
    Args:
        layer1_states: List of Layer 1 cached states (one per query)
        config: Layer2Config instance
        output_dir: Directory to save results
        api_manager_solve: The API manager for generating solutions
        api_manager_eval: The API manager for checking answers
    
    Returns:
        Tuple of (all_results, master_report)
    """
    orchestrator = Layer2Orchestrator(config, output_dir, api_manager_solve, api_manager_eval)
    
    logger.info(f"Starting Layer 2 Analysis on {len(layer1_states)} queries")
    logger.info(f"Config: {asdict(config)}")
    
    for layer1_state in layer1_states:
        orchestrator.run_single_query(layer1_state)
    
    # Generate and save reports (JSON and CSV)
    report = orchestrator.generate_master_report()
    orchestrator.save_reports(report)
    
    logger.info(f"Layer 2 Analysis Complete: {len(orchestrator.all_results)} total experiments")
    
    return orchestrator.all_results, report
