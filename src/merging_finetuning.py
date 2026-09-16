"""QLoRA preparation, training, binary-tree inference, and evaluation helpers.

Heavy Hugging Face and CUDA dependencies are imported only by the functions that
need them. Dataset parsing and tree orchestration therefore remain unit-testable
on an ordinary CPU environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evaluation import evaluate_single_answer_with_llm
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    create_merging_prompt,
)
from src.utils import save_json_atomic


_EXAMPLE_RE = re.compile(
    r"<Example\s+(\d+)>\s*Question:\s*(.*?)\s*"
    r"Rationale and Answer:\s*(.*?)\s*</Example\s+\1>",
    re.DOTALL | re.IGNORECASE,
)
_MAIN_QUESTION_RE = re.compile(
    r"<Main Question to Solve>\s*(.*?)\s*</Main Question to Solve>",
    re.DOTALL | re.IGNORECASE,
)


def normalize_question(text: str) -> str:
    """Return the stable identity used for deduplication and grouped splits."""
    return " ".join(str(text).split()).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_legacy_merging_prompt(input_prompt: str) -> Dict[str, Any]:
    """Parse the exact two-example ``final_solver_v3`` merging-data contract."""
    if not isinstance(input_prompt, str) or not input_prompt.strip():
        raise ValueError("input_prompt must be a non-empty string")
    main_matches = _MAIN_QUESTION_RE.findall(input_prompt)
    examples = _EXAMPLE_RE.findall(input_prompt)
    if len(main_matches) != 1:
        raise ValueError(f"expected one main question, found {len(main_matches)}")
    if len(examples) != 2:
        raise ValueError(f"expected exactly two examples, found {len(examples)}")
    main_question = main_matches[0].strip()
    examples.sort(key=lambda item: int(item[0]))
    indices = [int(item[0]) for item in examples]
    if indices != [1, 2]:
        raise ValueError(f"expected Example 1 and Example 2, found {indices}")
    candidate_questions = [item[1].strip() for item in examples]
    if any(normalize_question(q) != normalize_question(main_question) for q in candidate_questions):
        raise ValueError("both examples must answer the main question")
    candidates = [item[2].strip() for item in examples]
    if any(not candidate for candidate in candidates):
        raise ValueError("candidate solutions must be non-empty")
    return {"question": main_question, "candidate_solutions": candidates}


def load_merging_datasets(paths: Sequence[os.PathLike[str] | str]) -> Dict[str, Any]:
    """Load, validate, convert, and exact-deduplicate merging dataset files."""
    if not paths:
        raise ValueError("At least one merging dataset path is required.")
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append({"path": str(path), "sha256": _sha256_file(path)})
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        for source_index, item in enumerate(payload):
            try:
                if not isinstance(item, Mapping):
                    raise ValueError("record must be an object")
                label = item.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("label must be a non-empty string")
                parsed = parse_legacy_merging_prompt(item.get("input_prompt", ""))
                identity = normalize_question(parsed["question"])
                duplicate_key = (
                    identity,
                    tuple(parsed["candidate_solutions"]),
                    label.strip(),
                )
                if duplicate_key in seen:
                    failures.append({
                        "source_path": str(path), "source_index": source_index,
                        "reason": "exact_duplicate",
                    })
                    continue
                seen.add(duplicate_key)
                records.append({
                    "record_id": hashlib.sha256(
                        (str(path) + ":" + str(source_index)).encode("utf-8")
                    ).hexdigest()[:16],
                    "question": parsed["question"],
                    "question_id": identity,
                    "candidate_solutions": parsed["candidate_solutions"],
                    "prompt": create_merging_prompt(
                        parsed["question"], parsed["candidate_solutions"]
                    ),
                    "label": label.strip(),
                    "metadata": dict(item.get("metadata") or {}),
                    "source_path": str(path),
                    "source_index": source_index,
                })
            except (KeyError, TypeError, ValueError) as exc:
                failures.append({
                    "source_path": str(path), "source_index": source_index,
                    "reason": str(exc),
                })
    if not records:
        raise ValueError("No valid merging records were loaded.")
    return {"records": records, "failures": failures, "sources": sources}


def grouped_split(
    records: Sequence[Mapping[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split whole normalized-question groups into train/validation/test."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for raw_record in records:
        record = dict(raw_record)
        question_id = record.get("question_id") or normalize_question(record["question"])
        record["question_id"] = question_id
        groups.setdefault(question_id, []).append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three distinct questions are required for non-empty splits.")
    random.Random(seed).shuffle(keys)
    train_count = max(1, int(len(keys) * ratios[0]))
    validation_count = max(1, int(len(keys) * ratios[1]))
    if train_count + validation_count >= len(keys):
        train_count = len(keys) - 2
        validation_count = 1
    boundaries = (train_count, train_count + validation_count)
    split_keys = {
        "train": keys[: boundaries[0]],
        "validation": keys[boundaries[0] : boundaries[1]],
        "test": keys[boundaries[1] :],
    }
    return {
        name: [record for key in selected for record in groups[key]]
        for name, selected in split_keys.items()
    }


def _chat_tokens(tokenizer: Any, prompt: str, label: str) -> Dict[str, List[int]]:
    user_messages = [{"role": "user", "content": prompt}]
    full_messages = user_messages + [{"role": "assistant", "content": label}]
    prompt_ids = tokenizer.apply_chat_template(
        user_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the generation-prompt prefix.")
    return {"prompt_ids": list(prompt_ids), "input_ids": list(full_ids)}


def tokenize_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer: Any,
    max_length: int = 4096,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Tokenize without truncation and mask every non-assistant target token."""
    tokenized: Dict[str, List[Dict[str, Any]]] = {}
    report: Dict[str, Any] = {"max_length": max_length, "splits": {}}
    for split_name in ("train", "validation", "test"):
        kept: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        lengths: List[int] = []
        for record in splits.get(split_name, []):
            tokens = _chat_tokens(tokenizer, record["prompt"], record["label"])
            total_length = len(tokens["input_ids"])
            lengths.append(total_length)
            if total_length > max_length:
                excluded.append({
                    "record_id": record.get("record_id"),
                    "question_id": record.get("question_id"),
                    "token_length": total_length,
                    "reason": "overlength",
                })
                continue
            prompt_length = len(tokens["prompt_ids"])
            kept.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": [1] * total_length,
                "labels": [-100] * prompt_length + tokens["input_ids"][prompt_length:],
                "record_id": record.get("record_id"),
            })
        tokenized[split_name] = kept
        report["splits"][split_name] = {
            "input_records": len(splits.get(split_name, [])),
            "kept_records": len(kept),
            "excluded": excluded,
            "token_length_min": min(lengths) if lengths else None,
            "token_length_max": max(lengths) if lengths else None,
            "token_length_mean": sum(lengths) / len(lengths) if lengths else None,
        }
        if not kept:
            raise ValueError(f"Required split {split_name!r} is empty after token filtering.")
    return tokenized, report


def prepare_merging_data(
    paths: Sequence[os.PathLike[str] | str],
    output_dir: os.PathLike[str] | str,
    tokenizer: Optional[Any] = None,
    max_length: int = 4096,
    seed: int = 42,
) -> Dict[str, Any]:
    """Prepare grouped data and persist a reproducibility manifest."""
    loaded = load_merging_datasets(paths)
    splits = grouped_split(loaded["records"], seed=seed)
    manifest: Dict[str, Any] = {
        "format_version": 1,
        "seed": seed,
        "split_ratios": [0.8, 0.1, 0.1],
        "sources": loaded["sources"],
        "parsing_failures": loaded["failures"],
        "split_membership": {
            name: [record["record_id"] for record in values]
            for name, values in splits.items()
        },
        "split_question_ids": {
            name: sorted({record["question_id"] for record in values})
            for name, values in splits.items()
        },
    }
    tokenized = None
    if tokenizer is not None:
        tokenized, token_report = tokenize_splits(splits, tokenizer, max_length)
        manifest["token_statistics"] = token_report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not save_json_atomic(manifest, str(output_path / "data_manifest.json")):
        raise OSError("Failed to save data_manifest.json")
    return {"splits": splits, "tokenized": tokenized, "manifest": manifest}


class CompletionOnlyCollator:
    """Pad pre-tokenized examples while keeping prompt and padding labels masked."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = int(math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        input_rows, mask_rows, label_rows = [], [], []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be configured")
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_rows.append(list(feature["input_ids"]) + [pad_id] * padding)
            mask_rows.append(list(feature["attention_mask"]) + [0] * padding)
            label_rows.append(list(feature["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


@dataclass
class QLoRAConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir: str = "/kaggle/working/merging-qwen3-4b-qlora"
    gpu_index: int = 0
    max_length: int = 4096
    epochs: float = 3.0
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42


def load_qlora_model(config: QLoRAConfig) -> Tuple[Any, Any]:
    """Load a single-GPU, 4-bit NF4 model and attach trainable LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU.")
    if config.gpu_index < 0 or config.gpu_index >= torch.cuda.device_count():
        raise ValueError(f"CUDA GPU index {config.gpu_index} is unavailable.")
    bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": config.gpu_index},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    ))
    return model, tokenizer


def smoke_test_training_step(model: Any, collator: Any, example: Mapping[str, Any]) -> float:
    """Run one real forward/backward step before the full Kaggle job."""
    import torch

    model.train()
    batch = {key: value.to(model.device) for key, value in collator([example]).items()}
    try:
        loss = model(**batch).loss
        loss.backward()
        value = float(loss.detach().cpu())
        model.zero_grad(set_to_none=True)
        return value
    except RuntimeError as exc:
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "QLoRA smoke step failed. Keep the recorded settings and explicitly "
            "adjust max_length, gradient checkpointing, or model choice before retrying."
        ) from exc


def train_qlora(
    model: Any,
    tokenizer: Any,
    tokenized_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    config: QLoRAConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> Any:
    """Train and save the best validation-loss adapter and reproducibility data."""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bf16 = bool(torch.cuda.is_bf16_supported())
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(list(tokenized_splits["train"])),
        eval_dataset=Dataset.from_list(list(tokenized_splits["validation"])),
        data_collator=CompletionOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    best_dir = output_dir / "best_adapter"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    with (output_dir / "dependency_versions.json").open("w", encoding="utf-8") as handle:
        import accelerate, bitsandbytes, datasets, peft, transformers
        json.dump({
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "datasets": datasets.__version__,
            "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
        }, handle, indent=2)
    return trainer


class LocalFusionGenerator:
    """Generate with either the disabled base model or the active trained adapter."""

    def __init__(self, model: Any, tokenizer: Any, max_input_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens

    def __call__(
        self, prompt: str, *, use_adapter: bool, seed: int, temperature: float,
        top_p: float, max_new_tokens: int,
    ) -> Dict[str, Any]:
        import torch

        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + max_new_tokens > self.max_input_tokens:
            return {
                "status": "CONTEXT_OVERFLOW", "text": "", "input_tokens": input_tokens,
                "error": (
                    f"input ({input_tokens}) + requested output ({max_new_tokens}) exceeds "
                    f"the configured context budget ({self.max_input_tokens})"
                ),
            }
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        adapter_context = (
            nullcontext()
            if use_adapter
            else self.model.disable_adapter()
            if hasattr(self.model, "disable_adapter")
            else nullcontext()
        )
        started = time.perf_counter()
        try:
            with adapter_context, torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = output[0, input_tokens:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            eos_id = self.tokenizer.eos_token_id
            ended = eos_id is not None and eos_id in generated.tolist()
            status = "SUCCESS" if text and (ended or len(generated) < max_new_tokens) else (
                "LENGTH_LIMIT" if text else "EMPTY"
            )
            return {
                "status": status, "text": text, "input_tokens": input_tokens,
                "output_tokens": int(len(generated)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            return {
                "status": "GENERATION_FAILED", "text": "", "input_tokens": input_tokens,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


def load_adapter_for_inference(
    adapter_path: str, base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    gpu_index: int = 0,
) -> Tuple[Any, Any]:
    """Reload a saved adapter over its 4-bit base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        ),
        torch_dtype=dtype,
        device_map={"": gpu_index},
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def retrieve_exemplars_cpu(
    question: str,
    exemplar_questions: Sequence[str],
    exemplar_solutions: Sequence[str],
    embedded_exemplars: Any,
    embedding_model: Any,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve on CPU while excluding every normalized exact question match."""
    import numpy as np

    if top_k <= 0 or top_k % 2:
        raise ValueError("Retrieved-example count must be a positive even integer.")
    if len(exemplar_questions) != len(exemplar_solutions):
        raise ValueError("Exemplar questions and solutions must align.")
    query_vector = np.asarray(embedding_model.encode([question]), dtype=np.float32)[0]
    matrix = np.asarray(embedded_exemplars, dtype=np.float32)
    if matrix.shape[0] != len(exemplar_questions):
        raise ValueError("Embedding rows must align with exemplar questions.")
    denominator = np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector), 1e-12)
    similarities = matrix.dot(query_vector) / denominator
    target_id = normalize_question(question)
    similarities[[normalize_question(item) == target_id for item in exemplar_questions]] = -np.inf
    eligible = int(np.isfinite(similarities).sum())
    if eligible < top_k:
        raise ValueError(f"Only {eligible} non-identical exemplars are available for top_k={top_k}.")
    indices = sorted(range(len(similarities)), key=lambda idx: (-float(similarities[idx]), idx))[:top_k]
    return [{
        "index": idx, "question": exemplar_questions[idx],
        "solution": exemplar_solutions[idx], "similarity": float(similarities[idx]),
    } for idx in indices]


def _generation_node(
    generator: Callable[..., Dict[str, Any]], node_id: str, kind: str, prompt: str,
    parents: Sequence[str], use_adapter: bool, seed: int, generation: Mapping[str, Any],
) -> Dict[str, Any]:
    result = generator(
        prompt, use_adapter=use_adapter, seed=seed,
        temperature=generation["temperature"], top_p=generation["top_p"],
        max_new_tokens=generation["max_new_tokens"],
    )
    return {
        "node_id": node_id, "kind": kind, "parents": list(parents), "prompt": prompt,
        "use_adapter": use_adapter, "seed": seed, **result,
    }


def run_merging_tree(
    question: str,
    generator: Callable[..., Dict[str, Any]],
    mode: str,
    retrieved_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    zero_shot_n: Optional[int] = None,
    generation: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
    fusion_use_adapter: bool = True,
    resume_trace: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a deterministic binary fusion tree and preserve its complete trace."""
    if mode not in {"zero_shot", "retrieved", "mixed"}:
        raise ValueError("mode must be zero_shot, retrieved, or mixed")
    defaults = {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 1024}
    defaults.update(dict(generation or {}))
    if zero_shot_n is None:
        zero_shot_n = 8 if mode == "zero_shot" else 4
    if mode in {"zero_shot", "mixed"} and zero_shot_n <= 0:
        raise ValueError("zero_shot_n must be positive for zero_shot and mixed modes")
    retrieved = list(retrieved_examples or [])
    if mode in {"retrieved", "mixed"}:
        if not retrieved:
            raise ValueError("retrieved_examples are required for this mode")
        if len(retrieved) % 2:
            raise ValueError("retrieved_examples must have an even length")
    cached = {node["node_id"]: dict(node) for node in (resume_trace or []) if node.get("status") == "SUCCESS"}
    trace: List[Dict[str, Any]] = []

    def create(node_id: str, kind: str, prompt: str, parents: Sequence[str], adapter: bool, offset: int):
        node = cached.get(node_id) or _generation_node(
            generator, node_id, kind, prompt, parents, adapter, seed + offset, defaults
        )
        trace.append(node)
        return node

    def stop_if_failed(nodes: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        failed = [node["node_id"] for node in nodes if node.get("status") != "SUCCESS"]
        if failed:
            return {"status": "INCOMPLETE", "root_solution": None, "failed_nodes": failed, "trace": trace}
        return None

    zero_nodes: List[Dict[str, Any]] = []
    if mode in {"zero_shot", "mixed"}:
        for index in range(zero_shot_n):
            zero_nodes.append(create(
                f"zero-0-{index}", "zero_shot",
                create_final_reasoning_prompt_simple(question, {"PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v3"}),
                [], False, index,
            ))
        failure = stop_if_failed(zero_nodes)
        if failure:
            return failure

    retrieved_nodes: List[Dict[str, Any]] = []
    if mode in {"retrieved", "mixed"}:
        for source_index, item in enumerate(retrieved):
            trace.append({
                "node_id": f"retrieved-example-{source_index}",
                "kind": "retrieved_exemplar",
                "parents": [],
                "status": "SOURCE",
                "question": item["question"],
                "text": item["solution"],
                "retrieval_index": item.get("index"),
                "similarity": item.get("similarity"),
            })
        for pair_index in range(0, len(retrieved), 2):
            pair = retrieved[pair_index: pair_index + 2]
            prompt = create_final_reasoning_prompt(question, [
                EXEMPLAR_FORMAT.format(question=item["question"], solution=item["solution"])
                for item in pair
            ], {"PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v3"})
            retrieved_nodes.append(create(
                f"retrieved-1-{pair_index // 2}", "retrieved_transfer", prompt,
                [f"retrieved-example-{pair_index}", f"retrieved-example-{pair_index + 1}"],
                False, 1000 + pair_index // 2,
            ))
        failure = stop_if_failed(retrieved_nodes)
        if failure:
            return failure

    def fuse_layer(nodes: List[Dict[str, Any]], layer: int, prefix: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        next_nodes: List[Dict[str, Any]] = []
        for index in range(0, len(nodes) - 1, 2):
            left, right = nodes[index], nodes[index + 1]
            node = create(
                f"{prefix}-{layer}-{index // 2}", "fusion",
                create_merging_prompt(question, [left["text"], right["text"]]),
                [left["node_id"], right["node_id"]], fusion_use_adapter,
                layer * 100 + index // 2,
            )
            next_nodes.append(node)
        if len(nodes) % 2:
            carried = dict(nodes[-1])
            carried["carried_to_layer"] = layer
            next_nodes.append(carried)
        return next_nodes, stop_if_failed(next_nodes)

    if mode == "mixed":
        zero_nodes, failure = fuse_layer(zero_nodes, 1, "zero-fusion")
        if failure:
            return failure
        current = zero_nodes + retrieved_nodes
        layer = 2
    elif mode == "zero_shot":
        current, layer = zero_nodes, 1
    else:
        current, layer = retrieved_nodes, 2
    if not current:
        raise ValueError("A tree requires at least one generated solution.")
    while len(current) > 1:
        current, failure = fuse_layer(current, layer, "fusion")
        if failure:
            return failure
        layer += 1
    return {
        "status": "SUCCESS", "mode": mode, "question": question,
        "root_node_id": current[0]["node_id"], "root_solution": current[0]["text"],
        "trace": trace, "generation": defaults,
    }


def compare_base_and_adapted_trees(
    question: str,
    generator: Callable[..., Dict[str, Any]],
    mode: str,
    **tree_kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Compare fusion arms while reusing identical generated leaf solutions."""
    base = run_merging_tree(
        question, generator, mode, fusion_use_adapter=False, **tree_kwargs
    )
    leaf_trace = [
        node for node in base.get("trace", [])
        if node.get("status") == "SUCCESS"
        and node.get("kind") in {"zero_shot", "retrieved_transfer"}
    ]
    adapted = run_merging_tree(
        question, generator, mode, fusion_use_adapter=True,
        resume_trace=leaf_trace, **tree_kwargs
    )
    return {"base": base, "adapted": adapted}


def run_direct_fusion(
    question: str,
    candidate_solutions: Sequence[str],
    generator: Callable[..., Dict[str, Any]],
    *,
    use_adapter: bool,
    seed: int = 42,
    generation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fuse one stored candidate pair, used by held-out accepted evaluation."""
    defaults = {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 1024}
    defaults.update(dict(generation or {}))
    prompt = create_merging_prompt(question, list(candidate_solutions))
    node = _generation_node(
        generator, "direct-fusion-0", "fusion", prompt,
        ["stored-candidate-0", "stored-candidate-1"], use_adapter, seed, defaults,
    )
    success = node.get("status") == "SUCCESS"
    return {
        "status": "SUCCESS" if success else "INCOMPLETE",
        "mode": "direct", "question": question,
        "root_node_id": node["node_id"] if success else None,
        "root_solution": node.get("text") if success else None,
        "trace": [node], "generation": defaults,
    }


def build_evaluation_populations(
    prepared: Mapping[str, Any],
    benchmark_questions: Sequence[str],
    benchmark_ground_truths: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build held-out accepted and non-accepted benchmark populations safely."""
    if len(benchmark_questions) != len(benchmark_ground_truths):
        raise ValueError("Benchmark questions and ground truths must align.")
    benchmark: Dict[str, Dict[str, Any]] = {}
    for index, (question, ground_truth) in enumerate(zip(benchmark_questions, benchmark_ground_truths)):
        key = normalize_question(question)
        if key in benchmark:
            raise ValueError(f"Duplicate normalized benchmark question at index {index}.")
        benchmark[key] = {"question": question, "ground_truth": ground_truth, "benchmark_index": index}
    splits = prepared["splits"]
    accepted_ids = {record["question_id"] for values in splits.values() for record in values}
    heldout: List[Dict[str, Any]] = []
    for record in splits["test"]:
        if record["question_id"] not in benchmark:
            raise ValueError(f"Held-out accepted question is absent from benchmark: {record['question'][:80]!r}")
        heldout.append({**dict(record), **benchmark[record["question_id"]]})
    remaining = [value for key, value in benchmark.items() if key not in accepted_ids]
    return {"heldout_accepted": heldout, "remaining_benchmark": remaining}


def evaluate_tree_trace(
    tree_result: Mapping[str, Any],
    ground_truth: str,
    evaluator_manager: Any,
    evaluator_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Judge successful generated nodes and compute transparent transition counts."""
    judged: Dict[str, Optional[bool]] = {}
    judge_status: Dict[str, str] = {}
    for node in tree_result.get("trace", []):
        if node.get("status") != "SUCCESS" or not node.get("text"):
            continue
        result = evaluate_single_answer_with_llm(
            node["text"], ground_truth, evaluator_manager, dict(evaluator_config)
        )
        judged[node["node_id"]] = result.get("is_correct") if result.get("status") == "SUCCESS" else None
        judge_status[node["node_id"]] = result.get("status", "UNKNOWN")
    transitions = {"corrections": 0, "regressions": 0, "unchanged": 0, "unknown": 0}
    by_id = {node["node_id"]: node for node in tree_result.get("trace", [])}
    for node in tree_result.get("trace", []):
        if node.get("kind") != "fusion":
            continue
        child = judged.get(node["node_id"])
        parents = [judged.get(parent) for parent in node.get("parents", []) if parent in by_id]
        if child is None or len(parents) != 2 or any(value is None for value in parents):
            transitions["unknown"] += 1
        elif child and not all(parents):
            transitions["corrections"] += 1
        elif not child and any(parents):
            transitions["regressions"] += 1
        else:
            transitions["unchanged"] += 1
    root_id = tree_result.get("root_node_id")
    return {
        "root_correct": judged.get(root_id),
        "evaluation_coverage": sum(value is not None for value in judged.values()) / max(1, len(judged)),
        "node_correctness": judged,
        "judge_status": judge_status,
        "transitions": transitions,
    }


def summarize_evaluated_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Create a compact summary while keeping judge failures unknown."""
    known = [run["evaluation"]["root_correct"] for run in runs if run.get("evaluation", {}).get("root_correct") is not None]
    total_nodes = sum(
        node.get("status") != "SOURCE"
        for run in runs for node in run.get("tree", {}).get("trace", [])
    )
    failed = sum(run.get("tree", {}).get("status") != "SUCCESS" for run in runs)
    transitions = {"corrections": 0, "regressions": 0, "unchanged": 0, "unknown": 0}
    for run in runs:
        for name, count in run.get("evaluation", {}).get("transitions", {}).items():
            if name in transitions:
                transitions[name] += int(count)
    return {
        "runs": len(runs), "evaluated_roots": len(known),
        "accuracy_on_evaluated": sum(known) / len(known) if known else None,
        "evaluation_coverage": len(known) / len(runs) if runs else 0.0,
        "incomplete_trees": failed, "generated_nodes": total_nodes,
        "parent_to_child_transitions": transitions,
        "input_tokens": sum(
            node.get("input_tokens", 0)
            for run in runs for node in run.get("tree", {}).get("trace", [])
        ),
        "output_tokens": sum(
            node.get("output_tokens", 0)
            for run in runs for node in run.get("tree", {}).get("trace", [])
        ),
        "elapsed_seconds": sum(
            node.get("elapsed_seconds", 0.0)
            for run in runs for node in run.get("tree", {}).get("trace", [])
        ),
    }
